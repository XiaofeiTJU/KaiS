import math
import time
import json
import sys
from algorithm.cMMAC import *
from algorithm.GPG import *
from env.platform import *
from env.env_run import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def flatten(list):
    return [y for x in list for y in x]


def calculate_reward(master1, master2, cur_done, cur_undone):
    weight = 1.0
    all_task = [float(cur_done[0] + cur_undone[0]), float(cur_done[1] + cur_undone[1])]
    fail_task = [float(cur_undone[0]), float(cur_undone[1])]
    reward = []
    # The ratio of requests that violate delay requirements
    task_fail_rate = []
    if all_task[0] != 0:
        task_fail_rate.append(fail_task[0] / all_task[0])
    else:
        task_fail_rate.append(0)

    if all_task[1] != 0:
        task_fail_rate.append(fail_task[1] / all_task[1])
    else:
        task_fail_rate.append(0)

    # The standard deviation of the CPU and memory usage
    standard_list = []
    use_rate1 = []
    use_rate2 = []
    for i in range(3):
        use_rate1.append(master1.node_list[i].cpu / master1.node_list[i].cpu_max)
        use_rate1.append(master1.node_list[i].mem / master1.node_list[i].mem_max)
        use_rate2.append(master2.node_list[i].cpu / master2.node_list[i].cpu_max)
        use_rate2.append(master2.node_list[i].mem / master2.node_list[i].mem_max)

    standard_list.append(np.std(use_rate1, ddof=1))
    standard_list.append(np.std(use_rate2, ddof=1))

    reward.append(math.exp(-task_fail_rate[0]) + weight * math.exp(-standard_list[0]))
    reward.append(math.exp(-task_fail_rate[1]) + weight * math.exp(-standard_list[1]))
    return reward


def to_grid_rewards(node_reward):
    return np.array(node_reward).reshape([-1, 1])


def execution(RUN_TIMES, BREAK_POINT, TRAIN_TIMES, CHO_CYCLE):
    ############ Set up according to your own needs  ###########
    # The parameters are set to support the operation of the program, and may not be consistent with the actual system
    vaild_node = 6  # Number of edge nodes available
    SLOT_TIME = 0.5  # Time of one slot
    MAX_TESK_TYPE = 12  # Number of tesk types
    POD_CPU = 15.0  # CPU resources required for a POD
    POD_MEM = 1.0  # Memory resources required for a POD
    # Resource demand coefficients for different types of services
    service_coefficient = [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4]
    # Parameters related to DRL
    epsilon = 0.5
    gamma = 0.9
    learning_rate = 1e-3
    action_dim = 7
    state_dim = 88
    node_input_dim = 24
    cluster_input_dim = 24
    hid_dims = [16, 8]
    output_dim = 8
    max_depth = 8
    entropy_weight_init = 1
    exec_cap = 24
    entropy_weight_min = 0.0001
    entropy_weight_decay = 1e-3
    # Parameters related to GPU
    worker_num_gpu = 0
    worker_gpu_fraction = 0.1
    #####################################################################
    ########### Init ###########
    record = []
    throughput_list = []
    sum_rewards = []
    achieve_num = []
    achieve_num_sum = []
    fail_num = []
    deploy_reward = []
    current_time = str(time.time())
    log_dir = "./log/{}/".format(current_time)
    all_rewards = []
    order_response_rate_episode = []
    episode_rewards = []
    record_all_order_response_rate = []
    sess = tf.Session()
    tf.set_random_seed(1)
    q_estimator = Estimator(sess, action_dim, state_dim, 2, scope="q_estimator", summaries_dir=log_dir)
    sess.run(tf.global_variables_initializer())
    replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
    policy_replay = policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
    saver = tf.compat.v1.train.Saver()
    global_step1 = 0
    global_step2 = 0
    all_task1 = get_all_task('./data/Task_1.csv')
    all_task2 = get_all_task('./data/Task_2.csv')

    config = tf.ConfigProto(device_count={'GPU': worker_num_gpu},
                            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=worker_gpu_fraction))
    sess = tf.Session(config=config)
    orchestrate_agent = OrchestrateAgent(sess, node_input_dim, cluster_input_dim, hid_dims, output_dim, max_depth,
                                         range(1, exec_cap + 1))
    exp = {'node_inputs': [], 'cluster_inputs': [], 'reward': [], 'wall_time': [], 'node_act_vec': [],
           'cluster_act_vec': []}

    for n_iter in np.arange(RUN_TIMES):
        ########### Initialize the setup and repeat the experiment many times ###########

        batch_reward = []
        cur_time = 0
        entropy_weight = entropy_weight_init
        order_response_rates = []

        pre_done = [0, 0]
        pre_undone = [0, 0]
        context = [1, 1]
        ############ Set up according to your own needs  ###########
        # The parameters here are set only to support the operation of the program, and may not be consistent with the actual system
        deploy_state = [[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]]

        # Create clusters based on the hardware resources you need
        node1_1 = Node(100.0, 4.0, [], [])  # (cpu, mem,...)
        node1_2 = Node(200.0, 6.0, [], [])
        node1_3 = Node(100.0, 8.0, [], [])
        node_list1 = [node1_1, node1_2, node1_3]

        node2_1 = Node(200.0, 8.0, [], [])
        node2_2 = Node(100.0, 2.0, [], [])
        node2_3 = Node(200.0, 6.0, [], [])
        node_list2 = [node2_1, node2_2, node2_3]
        # (cpu, mem,..., achieve task num, give up task num)
        master1 = Master(200.0, 8.0, node_list1, [], all_task1, 0, 0, 0, [0] * MAX_TESK_TYPE, [0] * MAX_TESK_TYPE)
        master2 = Master(200.0, 8.0, node_list2, [], all_task2, 0, 0, 0, [0] * MAX_TESK_TYPE, [0] * MAX_TESK_TYPE)
        cloud = Cloud([], [], sys.maxsize, sys.maxsize)  # (..., cpu, mem)
        ################################################################################################
        for i in range(MAX_TESK_TYPE):
            docker = Docker(POD_MEM * service_coefficient[i], POD_CPU * service_coefficient[i], cur_time, i, [-1])
            cloud.service_list.append(docker)

        # Crerate dockers based on deploy_state
        for i in range(vaild_node):
            for ii in range(MAX_TESK_TYPE):
                dicision = deploy_state[i][ii]
                if i < 3 and dicision == 1:
                    j = i
                    if master1.node_list[j].mem >= POD_MEM * service_coefficient[ii]:
                        docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,
                                        ii, [-1])
                        master1.node_list[j].mem = master1.node_list[j].mem - POD_MEM * service_coefficient[ii]
                        master1.node_list[j].service_list.append(docker)

                if i >= 3 and dicision == 1:
                    j = i - 3
                    if master2.node_list[j].mem >= POD_MEM * service_coefficient[ii]:
                        docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,
                                        ii, [-1])
                        master2.node_list[j].mem = master2.node_list[j].mem - POD_MEM * service_coefficient[ii]
                        master2.node_list[j].service_list.append(docker)

        ########### Each slot ###########
        for slot in range(BREAK_POINT):
            cur_time = cur_time + SLOT_TIME
            ########### Each frame ###########
            if slot % CHO_CYCLE == 0 and slot != 0:
                done_tasks = []
                undone_tasks = []
                curr_tasks_in_queue = []
                # Get task state, include successful, failed, and unresolved
                for i in range(MAX_TESK_TYPE):
                    done_tasks.append(float(master1.done_kind[i] + master2.done_kind[i]))
                    undone_tasks.append(float(master1.undone_kind[i] + master2.undone_kind[i]))
                for i in range(3):
                    tmp = [0.0] * MAX_TESK_TYPE
                    for j in range(len(master1.node_list[i].task_queue)):
                        tmp[master1.node_list[i].task_queue[j][0]] = tmp[master1.node_list[i].task_queue[j][0]] + 1.0
                    curr_tasks_in_queue.append(tmp)
                for i in range(3):
                    tmp = [0.0] * MAX_TESK_TYPE
                    for k in range(len(master2.node_list[i].task_queue)):
                        tmp[master2.node_list[i].task_queue[k][0]] = tmp[master2.node_list[i].task_queue[k][0]] + 1
                    curr_tasks_in_queue.append(tmp)
                if slot != CHO_CYCLE:
                    exp['reward'].append(float(sum(deploy_reward)) / float(len(deploy_reward)))
                    deploy_reward = []
                    exp['wall_time'].append(cur_time)

                deploy_state_float = []
                for i in range(len(deploy_state)):
                    tmp = []
                    for j in range(len(deploy_state[0])):
                        tmp.append(float(deploy_state[i][j]))
                    deploy_state_float.append(tmp)
                    # Make decision of orchestration

                change_node, change_service, exp = act_offload_agent(orchestrate_agent, exp, done_tasks,
                                                                     undone_tasks, curr_tasks_in_queue,
                                                                     deploy_state_float)

                # Execute orchestration
                for i in range(len(change_node)):
                    if change_service[i] < 0:
                        # Delete docker and free memory
                        service_index = -1 * change_service[i] - 1
                        if change_node[i] < 3:
                            docker_idx = 0
                            while docker_idx < len(master1.node_list[change_node[i]].service_list):
                                if docker_idx >= len(master1.node_list[change_node[i]].service_list):
                                    break
                                if master1.node_list[change_node[i]].service_list[docker_idx].kind == service_index:
                                    master1.node_list[change_node[i]].mem = master1.node_list[change_node[i]].mem + \
                                                                            master1.node_list[
                                                                                change_node[i]].service_list[
                                                                                docker_idx].mem
                                    del master1.node_list[change_node[i]].service_list[docker_idx]
                                    deploy_state[change_node[i]][service_index] = deploy_state[change_node[i]][
                                                                                      service_index] - 1.0
                                else:
                                    docker_idx = docker_idx + 1
                        else:
                            node_index = change_node[i] - 3
                            docker_idx = 0
                            while docker_idx < len(master2.node_list[node_index].service_list):
                                if docker_idx >= len(master2.node_list[node_index].service_list):
                                    break
                                if master2.node_list[node_index].service_list[docker_idx].kind == service_index:
                                    master2.node_list[node_index].mem = master2.node_list[node_index].mem + \
                                                                        master2.node_list[node_index].service_list[
                                                                            docker_idx].mem
                                    del master2.node_list[node_index].service_list[docker_idx]
                                    deploy_state[node_index][service_index] = deploy_state[node_index][
                                                                                  service_index] - 1.0
                                else:
                                    docker_idx = docker_idx + 1
                    else:
                        # Add docker and tack up memory
                        service_index = change_service[i] - 1
                        if change_node[i] < 3:
                            if master1.node_list[change_node[i]].mem >= POD_MEM * service_coefficient[service_index]:
                                docker = Docker(POD_MEM * service_coefficient[service_index],
                                                POD_CPU * service_coefficient[service_index],
                                                cur_time, service_index, [-1])
                                master1.node_list[change_node[i]].mem = master1.node_list[
                                                                            change_node[i]].mem - POD_MEM * \
                                                                        service_coefficient[service_index]
                                master1.node_list[change_node[i]].service_list.append(docker)
                                deploy_state[change_node[i]][service_index] = deploy_state[change_node[i]][
                                                                                  service_index] + 1
                        else:
                            node_index = change_node[i] - 3
                            if master2.node_list[node_index].mem >= POD_MEM * service_coefficient[service_index]:
                                docker = Docker(POD_MEM * service_coefficient[service_index],
                                                POD_CPU * service_coefficient[service_index],
                                                cur_time, service_index, [-1])
                                master2.node_list[node_index].mem = master2.node_list[node_index].mem - POD_MEM * \
                                                                    service_coefficient[service_index]
                                master2.node_list[node_index].service_list.append(docker)
                                deploy_state[node_index][service_index] = deploy_state[node_index][service_index] + 1

                # Save data
                if slot > 3 * CHO_CYCLE:
                    exp_tmp = exp
                    del exp_tmp['node_inputs'][-1]
                    del exp_tmp['cluster_inputs'][-1]
                    del exp_tmp['node_act_vec'][-1]
                    del exp_tmp['cluster_act_vec'][-1]
                    entropy_weight, loss = train_orchestrate_agent(orchestrate_agent, exp_tmp, entropy_weight,
                                                                   entropy_weight_min, entropy_weight_decay)
                    entropy_weight = decrease_var(entropy_weight,
                                                  entropy_weight_min, entropy_weight_decay)

            # Get current task
            master1 = update_task_queue(master1, cur_time, 0)
            master2 = update_task_queue(master2, cur_time, 1)
            task1 = [-1]
            task2 = [-1]
            if len(master1.task_queue) != 0:
                task1 = master1.task_queue[0]
                del master1.task_queue[0]
            if len(master2.task_queue) != 0:
                task2 = master2.task_queue[0]
                del master2.task_queue[0]
            curr_task = [task1, task2]
            ava_node = []

            for i in range(len(curr_task)):
                tmp_list = [6]  # Cloud computing
                for ii in range(len(deploy_state)):
                    if deploy_state[ii][curr_task[i][0]] == 1:
                        tmp_list.append(ii)
                ava_node.append(tmp_list)

            # Current state of CPU and memory
            cpu_list1 = []
            mem_list1 = []
            cpu_list2 = []
            mem_list2 = []
            task_num1 = [len(master1.task_queue)]
            task_num2 = [len(master2.task_queue)]
            for i in range(3):
                cpu_list1.append([master1.node_list[i].cpu, master1.node_list[i].cpu_max])
                mem_list1.append([master1.node_list[i].mem, master1.node_list[i].mem_max])
                task_num1.append(len(master1.node_list[i].task_queue))
            for i in range(3):
                cpu_list2.append([master2.node_list[i].cpu, master2.node_list[i].cpu_max])
                mem_list2.append([master2.node_list[i].mem, master2.node_list[i].mem_max])
                task_num2.append(len(master2.node_list[i].task_queue))
            s_grid = np.array([flatten(flatten([deploy_state, [task_num1], cpu_list1, mem_list1])),
                               flatten(flatten([deploy_state, [task_num2], cpu_list1, mem_list1]))])

            # Dispatch decision
            act, valid_action_prob_mat, policy_state, action_choosen_mat, \
            curr_state_value, curr_neighbor_mask, next_state_ids = q_estimator.action(s_grid, ava_node, context,
                                                                                      epsilon)
            # Put the current task on the queue based on dispatch decision
            for i in range(len(act)):
                if curr_task[i][0] == -1:
                    continue
                if act[i] == 6:
                    cloud.task_queue.append(curr_task[i])
                    continue
                if act[i] >= 0 and act[i] < 3:
                    master1.node_list[act[i]].task_queue.append(curr_task[i])
                    continue
                if act[i] >= 3 and act[i] < 6:
                    master2.node_list[act[i] - 3].task_queue.append(curr_task[i])
                    continue
                else:
                    pass
            # Update state of task
            for i in range(3):
                master1.node_list[i].task_queue, undone, undone_kind = check_queue(master1.node_list[i].task_queue,
                                                                                   cur_time)
                for j in undone_kind:
                    master1.undone_kind[j] = master1.undone_kind[j] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]

                master2.node_list[i].task_queue, undone, undone_kind = check_queue(master2.node_list[i].task_queue,
                                                                                   cur_time)
                for j in undone_kind:
                    master2.undone_kind[j] = master2.undone_kind[j] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]

            cloud.task_queue, undone, undone_kind = check_queue(cloud.task_queue, cur_time)
            master1.undone = master1.undone + undone[0]
            master2.undone = master2.undone + undone[1]

            # Update state of dockers in every node
            for i in range(3):
                master1.node_list[i], undone, done, done_kind, undone_kind = update_docker(master1.node_list[i],
                                                                                           cur_time,
                                                                                           service_coefficient, POD_CPU)
                for j in range(len(done_kind)):
                    master1.done_kind[done_kind[j]] = master1.done_kind[done_kind[j]] + 1
                for j in range(len(undone_kind)):
                    master1.undone_kind[undone_kind[j]] = master1.undone_kind[undone_kind[j]] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]
                master1.done = master1.done + done[0]
                master2.done = master2.done + done[1]

                master2.node_list[i], undone, done, done_kind, undone_kind = update_docker(master2.node_list[i],
                                                                                           cur_time,
                                                                                           service_coefficient, POD_CPU)
                for j in range(len(done_kind)):
                    master1.done_kind[done_kind[j]] = master1.done_kind[done_kind[j]] + 1
                for j in range(len(undone_kind)):
                    master1.undone_kind[undone_kind[j]] = master1.undone_kind[undone_kind[j]] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]
                master1.done = master1.done + done[0]
                master2.done = master2.done + done[1]

            cloud, undone, done, done_kind, undone_kind = update_docker(cloud, cur_time, service_coefficient, POD_CPU)
            master1.undone = master1.undone + undone[0]
            master2.undone = master2.undone + undone[1]
            master1.done = master1.done + done[0]
            master2.done = master2.done + done[1]

            cur_done = [master1.done - pre_done[0], master2.done - pre_done[1]]
            cur_undone = [master1.undone - pre_undone[0], master2.undone - pre_undone[1]]

            pre_done = [master1.done, master2.done]
            pre_undone = [master1.undone, master2.undone]

            achieve_num.append(sum(cur_done))
            fail_num.append(sum(cur_undone))
            immediate_reward = calculate_reward(master1, master2, cur_done, cur_undone)

            record.append([master1, master2, cur_done, cur_undone, immediate_reward])

            deploy_reward.append(sum(immediate_reward))

            if slot != 0:
                r_grid = to_grid_rewards(immediate_reward)
                targets_batch = q_estimator.compute_targets(action_mat_prev, s_grid, r_grid, gamma)

                # Advantage for policy network.
                advantage = q_estimator.compute_advantage(curr_state_value_prev, next_state_ids_prev,
                                                          s_grid, r_grid, gamma)
                if curr_task[0][0] != -1 and curr_task[1][0] != -1:
                    replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)
                    policy_replay.add(policy_state_prev, action_choosen_mat_prev, advantage, curr_neighbor_mask_prev)

            # For updating
            state_mat_prev = s_grid
            action_mat_prev = valid_action_prob_mat
            action_choosen_mat_prev = action_choosen_mat
            curr_neighbor_mask_prev = curr_neighbor_mask
            policy_state_prev = policy_state

            # for computing advantage
            curr_state_value_prev = curr_state_value
            next_state_ids_prev = next_state_ids
            global_step1 += 1
            global_step2 += 1

            all_rewards.append(sum(immediate_reward))
            batch_reward.append(immediate_reward)

            if (sum(cur_done) + sum(cur_undone)) != 0:
                order_response_rates.append(float(sum(cur_done) / (sum(cur_done) + sum(cur_undone))))
            else:
                order_response_rates.append(0)

        sum_rewards.append(float(sum(all_rewards)) / float(len(all_rewards)))
        all_rewards = []

        all_number = sum(achieve_num) + sum(fail_num)
        throughput_list.append(sum(achieve_num) / float(all_number))
        print('throughput_list_all =', throughput_list, '\ncurrent_achieve_number =', sum(achieve_num),
              ', current_fail_number =', sum(fail_num))
        achieve_num = []
        fail_num = []

        episode_reward = np.sum(batch_reward[1:])
        episode_rewards.append(episode_reward)
        n_iter_order_response_rate = np.mean(order_response_rates[1:])
        order_response_rate_episode.append(n_iter_order_response_rate)
        record_all_order_response_rate.append(order_response_rates)
        # update value network
        for _ in np.arange(TRAIN_TIMES):
            batch_s, _, batch_r, _ = replay.sample()
            q_estimator.update_value(batch_s, batch_r, 1e-3, global_step1)
            global_step1 += 1

        # update policy network
        for _ in np.arange(TRAIN_TIMES):
            batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
            q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,
                                      global_step2)
            global_step2 += 1
        saver.save(sess, "./model/model.ckpt")
    saver.save(sess, "./model/model_before_testing.ckpt")
    tf.reset_default_graph()
    time_str = str(time.time())
    with open("./result/" + time_str + ".json", "w") as f:
        json.dump(record, f)
    return throughput_list


if __name__ == "__main__":
    ############ Set up according to your own needs  ###########
    # The parameters are set to support the operation of the program, and may not be consistent with the actual system
    RUN_TIMES = 500
    TASK_NUM = 5000
    TRAIN_TIMES = 50
    CHO_CYCLE = 1000
    ##############################################################
    execution(RUN_TIMES, TASK_NUM, TRAIN_TIMES, CHO_CYCLE)

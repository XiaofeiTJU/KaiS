import csv


def get_all_task(path):
    type_list = []
    start_time = []
    end_time = []
    cpu_list = []
    mem_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            type_list.append(row[3])
            start_time.append(row[5])
            end_time.append(row[6])
            cpu_list.append(row[7])
            mem_list.append(row[8])

    init_time = int(start_time[0])
    for i in range(len(start_time)):
        type_list[i] = int(type_list[i]) - 1
        start_time[i] = int(start_time[i]) - init_time
        end_time[i] = int(end_time[i]) - init_time
        cpu_list[i] = int(cpu_list[i]) / 100.0
        mem_list[i] = float(mem_list[i])
    all_task = [type_list, start_time, end_time, cpu_list, mem_list]

    return all_task


def put_task(task_queue, task):
    for i in range(len(task_queue) - 1):
        j = len(task_queue) - i - 1
        task_queue[j] = task_queue[j - 1]
    task_queue[0] = task
    return task_queue


def update_task_queue(master, cur_time, master_id):
    # clean task for overtime
    i = 0
    while len(master.task_queue) != i:
        if master.task_queue[i][0] == -1:
            i = i + 1
            continue
        if cur_time >= master.task_queue[i][2]:
            del master.task_queue[i]
            master.undone = master.undone + 1
            master.undone_kind[master.task_queue[i][0]] = master.undone_kind[master.task_queue[i][0]] + 1
        else:
            i = i + 1
    # get new task
    while master.all_task[1][master.all_task_index] < cur_time:
        task = [master.all_task[0][master.all_task_index], master.all_task[1][master.all_task_index],
                master.all_task[2][master.all_task_index], master.all_task[3][master.all_task_index],
                master.all_task[4][master.all_task_index], master_id]
        master.task_queue.append(task)
        master.all_task_index = master.all_task_index + 1

    tmp_list = []
    for i in range(len(master.task_queue)):
        if master.task_queue[i][0] != -1:
            tmp_list.append(master.task_queue[i])
    tmp_list = sorted(tmp_list, key=lambda x: (x[2], x[1]))
    master.task_queue = tmp_list
    return master


def check_queue(task_queue, cur_time):
    task_queue = sorted(task_queue, key=lambda x: (x[2], x[1]))
    undone = [0, 0]
    undone_kind = []
    # clean task for overtime
    i = 0
    while len(task_queue) != i:
        flag = 0
        if cur_time >= task_queue[i][2]:
            undone[task_queue[i][5]] = undone[task_queue[i][5]] + 1
            undone_kind.append(task_queue[i][0])
            del task_queue[i]
            flag = 1
        if flag == 1:
            flag = 0
        else:
            i = i + 1
    return task_queue, undone, undone_kind


def update_docker(node, cur_time, service_coefficient, POD_CPU):
    done = [0, 0]
    undone = [0, 0]
    done_kind = []
    undone_kind = []

    # find achieved task in current time
    for i in range(len(node.service_list)):
        if node.service_list[i].available_time <= cur_time and len(node.service_list[i].doing_task) > 1:
            done[node.service_list[i].doing_task[5]] = done[node.service_list[i].doing_task[5]] + 1
            done_kind.append(node.service_list[i].doing_task[0])
            node.service_list[i].doing_task = [-1]
            node.service_list[i].available_time = cur_time
    # execute task in queue
    i = 0
    while i != len(node.task_queue):
        flag = 0
        for j in range(len(node.service_list)):
            if i == len(node.task_queue):
                break
            if node.task_queue[i][0] == node.service_list[j].kind:
                if node.service_list[j].available_time > cur_time:
                    continue
                if node.service_list[j].available_time <= cur_time:
                    to_do = (node.task_queue[i][3]) / node.service_list[j].cpu
                    if cur_time + to_do <= node.task_queue[i][2] and node.cpu >= POD_CPU * service_coefficient[
                        node.task_queue[i][0]]:
                        node.cpu = node.cpu - POD_CPU * service_coefficient[node.task_queue[i][0]]
                        node.service_list[j].available_time = cur_time + to_do
                        node.service_list[j].doing_task = node.task_queue[i]
                        del node.task_queue[i]
                        flag = 1

                    elif cur_time + to_do > node.task_queue[i][2]:
                        undone[node.task_queue[i][5]] = undone[node.task_queue[i][5]] + 1
                        undone_kind.append(node.task_queue[i][0])
                        del node.task_queue[i]
                        flag = 1
                    elif node.cpu < POD_CPU * service_coefficient[node.task_queue[i][0]]:
                        pass

        if flag == 1:
            flag = 0
        else:
            i = i + 1
    return node, undone, done, done_kind, undone_kind

# A Simulator Demo to Demonstrate the Scheduling Process of *KaiS*

In order to provide interested readers with an intuitive understanding of the *KaiS* scheduling framework, especially the scheduling algorithms, we have extracted  the code of the cluster system, handcrafted a simple simulated environment (for just indication), exposed a part of data about service requests, and made appropriate adjustments to the proposed algorithms.
Interested readers can thus perceive the details of the implementation of our scheduling algorithms.
Hope this project can provide or inspire a promising and practical solution for intelligent cloud-edge cluster scheduling.

> Tailored Learning-Based Scheduling of Kubernetes-Oriented Edge-Cloud System, INFOCOM 2021. 
> Yiwen Han, Shihao Shen, Xiaofei Wang, Shiqiang Wang, Victor C. M. Leung

### Prerequisites

The code runs on Python 3 with TensorFlow. To install the dependencies, please execute:

```
pip3 install -r requirements.txt
```

### Project

- algorithm - includes the relevant scheduling algorithms, i.e., *cMMAC* and *GPG*
- data - includes the handcrafted service requests derived from Alibaba traces (https://github.com/alibaba/clusterdata)
- env - includes the code of simulating the edge-cloud cluster system
- log - includes the log of this program
- results - includes numerical changes in performance indicators (e.g., the throughput rate), noted that the simulation results may not be exactly the same as in the practical systems used in the paper

### Trace

As described in the paper, we use the Alibaba trace as shown below.

> Alibaba Cluster Trace Program
> https://github.com/alibaba/clusterdata

### Getting Started

* Trace: You can download the required data from the above Alibaba trace. At the same time, we also provide the data sample in ./data.
* Parameter setting: Set up in main.py according to your own needs.
* Run:  ` main.py`
* Observe: Read and analyze the json files in "\results".

### Version
* 0.1 beta

### Citation

If this paper can benefit your scientific publications, please kindly cite it along with the data provided by Alibaba.
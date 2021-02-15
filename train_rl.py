from env.Environment import Environment, State
from algo.algo import ValueIterationRL
from network.network import PatientClassificationNet, PatientGroupNet
from DataProcess import DataProcess
import torch
import pickle


def main():
    net = load_net(21, 256, [0.0216, 0.0470, 0.0711, 0.1185, 1.], './model/patient_group_net.pth')
    p_arrive, p_leave, delta, _ = DataProcess(net, time_interval=10).process()
    delta = [0] + delta
    # p_arrive = [0.8988013698630137, 0.012994672754946727, 0.01293759512937595, 0.01221461187214612, 0.013032724505327244, 0.013089802130898021, 0.011910197869101979, 0.012271689497716894, 0.012747336377473363]
    # p_leave = [0.010825796872474158, 0.0038442559604472373, 0.00335442100653245, 0.0027426663004428303, 0.0026808956939254076, 0.0026285227575753496, 0.002433672892274736, 0.0018672590670831529]
    # delta = [0, 0.00749376480848956, 0.023252303608424292, 0.38966471920753987, 0.1, 0.04492930254495119, 0.019348297623417, 1.1000634771910827, 1.080419000471227]
    print(p_arrive, p_leave, delta)
    env = Environment(5, 30, p_arrive, p_leave, delta, max_leave=1)
    env.calc_dynamics()
    rl = ValueIterationRL(env)
    x, a = rl.value_iteration_sparse()
    save_env(env)


def load_net(input_dim, hidden, class_boundary, path):
    class_boundary = torch.Tensor(class_boundary)
    temp = PatientClassificationNet(input_dim, hidden)
    net = PatientGroupNet(temp, class_boundary)
    net.load_state_dict(torch.load(path))
    return net


def save_env(env):
    filename = './log/env.pkl'
    with open(filename, 'wb') as output:
        pickle.dump(env, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
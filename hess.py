import torch
import numpy as np

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, _ = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph, allow_unused=True)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)

def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def select_param(param, prob):
    result = []
    if len(list(param.size())) > 1:
        for i in param:
            result_tmp = select_param(i, prob)
            if len(result_tmp):
                    result.extend(result_tmp)
    else:
        for i in param:
            p = np.random.binomial(1, prob)
            if p == 1:
                # print(i)
                # print(list(i.size()))
                result.extend([i])
    return result


def hutchinson(args, net, loss_super, outputs, device):

    params = [outputs]
    trace = 0.
    jacobi_norm = 0
    hessian_tr = 0
    grad_list = []


    for name, param in net.named_parameters():
        if param.requires_grad:
            # param= param.reshape(-1)
            # p = np.random.binomial(1, args.prob)
            # if p == 1:
            if 'weight' in name:
                if args.prob == 1:
                    params.append(param)
                else:
                    p = np.random.binomial(1, prob)
                    if p == 1:
                        params.append(param)
    # print(params)
    # params = torch.tensor(params)
    grads = torch.autograd.grad(loss_super, params, retain_graph=True, create_graph=True)

    for i in grads:
        grad_list.append(i)

    # calculate hessian trace
    trace_vhv = []

    if len(grad_list) > 0:
        for iii in range(args.Hiter):
            v = [torch.randint_like(p, high=1, device=device) for p in params]
            if args.prob == 1:
                for v_i in v:
                    v_i[v_i == 0] = -1
            else:
                for v_i in v:
                    v_i[v_i == 0] = np.random.binomial(1, args.prob * 2)
                for v_i in v:
                    v_i[v_i == 1] = 2 * np.random.binomial(1, 0.5) - 1
            Hv = torch.autograd.grad(grad_list,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=True)
            # hessian_tr = torch.trace(hess)
            hessian_tr = group_product(Hv, v).cpu().item()
            #trace_vhv.append(hessian_tr)
            trace += hessian_tr

    return trace, hessian_tr
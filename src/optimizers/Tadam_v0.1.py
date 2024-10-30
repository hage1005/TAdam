import torch
import wandb
class Tadam(torch.optim.Optimizer):
    def __init__(self, params, total_steps, lr=1e-3, beta1=0.9, beta2=0.999, gamma=0.25, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, gamma=gamma, eps=eps)
        super(Tadam, self).__init__(params, defaults)
        self.total_steps = total_steps
        self.t = 1.0  # timestep
        self.ls = 0.0  # loss
        self.ls_h = 0.0  # loss_hat
        self.pr = 0.0  # predict reduction
        self.dt = 1.0  # delta
        self.bp1 = beta1  # bias correction 1
        self.bp2 = beta2  # bias correction 2

        # Initialize the state variables for all parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['m'] = torch.zeros_like(p.data)
                    state['m_h'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data)
                    state['v'] = torch.ones_like(p.data) * eps

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if loss is None:
            return

        if self.t < 1.1:
            delta = 1.0
        else:
            delta = self.dt


        bp1, bp2 = self.bp1, self.bp2
        bc1 = 1.0 - bp1
        bc2 = 1.0 - bp2
        beta1, beta2 = self.defaults['beta1'], self.defaults['beta2']
        
        self.ls = beta1 * self.ls + (1.0 - beta1) * loss.item()
        self.ls_h = self.ls / bc1

        pr_temp = 0.0

        for group in self.param_groups:
            beta1, beta2, gamma, eps = group['beta1'], group['beta2'], group['gamma'], group['eps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                m, m_h = state['m'], state['m_h']

                s, v = state['s'], state['v']

                dv = ((p.grad.data - m_h) ** 2) * (beta2 - bp2) / bc2
                state['v'] = beta2 * v + (1.0 - beta2) * dv
                v_h = state['v'] / bc2

                state['m'] = beta1 * m + (1.0 - beta1) * p.grad.data
                state['m_h'] = state['m'] / bc1

                state['s'] = beta2 * s + (1.0 - beta2) * (p.grad.data ** 2)
                s_h = state['s'] / bc2

                f_h = (1.0 + torch.sum((m_h ** 2) / (v_h + eps))) * v_h

                u_h = torch.maximum(delta * f_h, torch.sqrt(s_h))
                g_h = m_h * delta / (u_h + eps)

                # Directly apply update without flattening
                p.data.add_(-lr * g_h)

                pr1 = torch.sum(m_h * g_h)
                pr2 = pr1 ** 2 + torch.sum(v_h * g_h ** 2)
                pr_temp += (pr1 - 0.5 * pr2) * lr

        self.bp1 = bp1 * beta1
        self.bp2 = bp2 * beta2
        self.t += 1.0
        self.pr = pr_temp
        self.dt = self.compute_delta(loss.item(), self.dt)

    def compute_delta(self, loss, dt):
        rho = (self.ls_h - loss) / max(self.pr, self.defaults['eps'])
        wandb.log({'rho': rho})
        dt_min = torch.pow(torch.tensor(1.0 - self.defaults['gamma']), (self.t - 1.0) / self.total_steps)
        dt_max = 1.0 + torch.pow(torch.tensor(self.defaults['gamma']), (self.t - 1.0) / self.total_steps)

        if rho < self.defaults['gamma']:
            new_dt = dt_min
        elif rho > (1.0 - self.defaults['gamma']):
            new_dt = dt_max
        else:
            new_dt = 1.0
        dt = torch.clamp(new_dt * dt, min=dt_min, max=dt_max)
        return dt

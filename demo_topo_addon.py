import torch
from lphp.loss import TopoLossCOD
from lphp.interfaces import topo_addon


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch, channels, height, width = 2, 1, 64, 64
    student_logits = torch.randn(batch, channels, height, width, device=device, requires_grad=True)
    teacher_logits = torch.randn(batch, channels, height, width, device=device)
    sod_loss = student_logits.pow(2).mean()

    topo = TopoLossCOD(
        lambda_ph=1.0,
        ph_mode='cap',
        ph_num_thresholds=16,
        ph_downsample=4,
        ph_max_events=12,
        ph_tgt_smooth_ks=3,
        ph_importance_mix='auto',
        ph_use_uncert_gate=False,
    ).to(device)
    topo.train()

    out = topo_addon(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        topo=topo,
        sod_loss_for_balance=sod_loss,
        lambda_topo=1e-1,
        iter_percentage=0.6,
        ramp_milestones=(0.3, 0.8),
        grad_balance=True,
        amp_device='cuda' if torch.cuda.is_available() else 'cpu',
        use_ortho=False,
    )

    if out is None:
        raise RuntimeError('topo_addon returned None')

    total = sod_loss + out.weighted_loss
    total.backward()

    print('weighted_loss =', float(out.weighted_loss.detach().cpu().item()))
    print('alpha =', out.alpha)
    print('topo_coef =', out.topo_coef)
    print('log_str =', out.log_str)
    print('parts keys =', sorted(k for k in out.parts.keys() if not k.startswith('_')))
    print('backward ok')


if __name__ == '__main__':
    main()

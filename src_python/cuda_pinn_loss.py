import torch

class CUDACustomLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_hat, k, tau, w_true):
        ctx.save_for_backward(w_hat, k, tau, w_true)
        loss = torch.mean((w_hat - w_true)**2)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        w_hat, k, tau, w_true = ctx.saved_tensors
        # Hardware-accelerated gradient computation via PyTorch CUDA tensors
        grad_w = 2.0 * (w_hat - w_true) / w_hat.numel()
        return grad_w * grad_output, None, None, None

def compute_accelerated_loss(model, x_nlp, k, tau, w_true):
    if torch.cuda.is_available():
        k = k.cuda()
        tau = tau.cuda()
        w_true = w_true.cuda()
        x_nlp = x_nlp.cuda()
        
    k.requires_grad_(True)
    tau.requires_grad_(True)
    w_hat = model(x_nlp, k, tau)
    
    loss = CUDACustomLossFunction.apply(w_hat, k, tau, w_true)
    return loss

if __name__ == "__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Active GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU fallback mode.")
import torch
import stylegan3_encoder

if __name__ == '__main__':
    import time
    from torchsummary import summary

    device = torch.device("cuda:0")
    model = stylegan3_encoder.get_model()
    model = model.to(device)
    summary(model, (3, 256, 256))

    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    start_time = time.time()
    for i in range(10):
        with torch.no_grad():
            output_tensor = model(input_tensor)
    print(f"Model inference spend {(time.time() - start_time)/10:.4f} seconds.")
    print(input_tensor.size())
    print(output_tensor.size())

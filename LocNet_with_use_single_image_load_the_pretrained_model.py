if __name__ == '__main__':
    from LocNet_with_train import LocNet
    import torch
    from torchvision.io import read_image, ImageReadMode
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--sampling_image",
        dest="SAMPLING_IMAGE",
        required=True,
        help="Path to a test image."
    )
    ap.add_argument(
        "-k",
        "--known_pixel_and_environment_buildings",
        dest="ENVIRONMENT_BUILDINGS",
        required=True,
        help="Path to the test environment buildings with known sampling pixel."
    )
    args = vars(ap.parse_args())
    sampling_data = read_image(args["SAMPLING_IMAGE"], ImageReadMode.GRAY).float() / 255.0
    environment_building = read_image(args["ENVIRONMENT_BUILDINGS"], ImageReadMode.GRAY) / 255.0
    # convert from C H W to B C H W
    data_input = torch.unsqueeze(torch.cat([sampling_data, environment_building]), dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LocNet(
        enc_in=2,
        enc_out=4,
        dec_out=1,
        n_dim=27,
        leaky_relu=0.3,
        outpath=None,
    )
    model = model.to(device)
    model.load_state_dict(torch.load('Pretrained_LocNet.pt', weights_only=True))
    data_input = data_input.to(device)
    output = model.forward(data_input).squeeze().detach().cpu()
    transmitter_prediction = torch.argmax(torch.flatten(output))
    # x is col and y is row
    print(f"Prediction => x coord: {transmitter_prediction % 256}, y coord: {transmitter_prediction // 256}")

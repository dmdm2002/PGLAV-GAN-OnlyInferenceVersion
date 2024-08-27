import argparse
import glob
import os
import re
import yaml

import torch
import torchvision.transforms as transforms
import PIL.Image as Image

from model.pgla_vit_generator import Gen


def get_configs(path):
    assert os.path.exists(path), f"경로[{path}]에 해당 파일이 존재하지 않습니다. 프로그램을 종료합니다."
    with open(path) as f:
        config = yaml.safe_load(f)

    return config

def check_output_folder(output_folder):
    if os.path.isdir(output_folder):
        print("폴더 확인 완료")
    else:
        print("저장 폴더가 존재하지 않아 생성합니다.")
        os.makedirs(output_folder, exist_ok=True)
        print("폴더 생성 완료")


def check_input_folder(input_folder):
    assert os.path.isdir(input_folder), "입력 폴더가 존재하지 않습니다. 프로그램을 종료합니다."


def to_tensor(img):
    transform_module = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

    return transform_module(img).unsqueeze(0)


def to_image(img):
    transform_module = transforms.Compose(
        [
            transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
            transforms.ToPILImage(),
        ]
    )

    return transform_module(img)


def save_image(output, img_path, output_folder, idx):
    output = to_image(output.squeeze())
    name = img_path.split("/")[-1]
    name = f'fake_{name}'
    output.save(f'{output_folder}/{name}')
    print(f"[위조 이미지 저장-{idx}] 경로: {output_folder}/{name}")
    print("========================================================================")


def inference(input_images, output_folder, device, config):
    device = torch.device(device)
    print(f'[Device Setting!!]: {device}')

    model = Gen(config=config).to(device)

    ckp_path = 'pglav_gan_ckp.pth'
    checkpoint = torch.load(ckp_path, map_location=device)
    model.load_state_dict(checkpoint['netG_A2B_state_dict'])
    model.eval()

    for idx, img_path in enumerate(input_images):
        img_path = re.compile("\\\\").sub("/", img_path)
        print(f'[위조 이미지 생성-{idx}] 원본 이미지 경로: {img_path}')
        img = Image.open(img_path)
        img = to_tensor(img).to(device)

        output = model(img)

        save_image(output, img_path, output_folder, idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='Input Image Folder')
    parser.add_argument('--output_folder', type=str, help='Output Image Folder')
    parser.add_argument('--device', type=str, default='cpu', help='Select Device, cpu or cuda')

    args = parser.parse_args()
    config = get_configs('configs/config.yml')

    check_input_folder(args.input_folder)
    input_images = glob.glob(f'{args.input_folder}/*')
    assert len(input_images), "이미지가 존재하지 않습니다!!"

    check_output_folder(args.output_folder)

    if args.device.lower() == 'cuda' and torch.cuda.is_available() is False:
        print('GPU를 찾을 수 없습니다. CPU로 device를 변경합니다.')
        args.device = 'cpu'

    print('위조 이미지 생성 시작')
    inference(input_images=input_images, output_folder=args.output_folder, device=args.device, config=config)
    print('위조 이미지 생성 종료')




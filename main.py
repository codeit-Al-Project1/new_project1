import argparse
from src.json_modify import json_modify


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Json modification")
    # default 저장 루트를 확인
    parser.add_argument('--output_dir', type=str, default="data/train_annots_modify", help='Destination of json file save')
    parser.add_argument('--json_folder', type=str, default='data/train_annotations', help='Location of original json files')
    args = parser.parse_args()
    json_modify(args.output_dir, args.json_folder)
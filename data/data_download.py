import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_data(download_path = './',download=True, extract=True):
    """
    Kaggle API를 이용해 데이터를 다운로드하고, 압축을 해제하는 메소드, 
    메소드의 마지막 과정에서 압축파일을 제거한다.

    ***Note: Kaggle API를 사용하기 위해서는 Kaggle 계정이 있어야 하며,
    Kaggle API Token이 필요하다.***
    ***Note: 다운로드 위치를 주의하도록 하자.***
    
    :param download_path: str: 다운로드 경로
    :param download: bool: 데이터 다운로드 여부
    :param extract: bool: 압축 해제 여부
    """

    competition_name = 'ai01-level1-project'

    if download:
        api = KaggleApi()
        api.authenticate()

        if f"{competition_name}.zip" in os.listdir(download_path):
            print(f"{competition_name}가 {download_path}에 이미 존재합니다.")
        else:
            api.competition_download_files(competition_name, path=download_path)
            if download_path == './':
                print(f"현재위치에 다운로드 완료!")
            else:
                print(f"다운로드 완료! 저장위치: {download_path}")

    if extract:
        zip_file = f'{competition_name}.zip'
        if zip_file not in os.listdir(download_path):
            print(f"{zip_file} not found in {download_path}")
            return
        
        zip_path = os.path.join(download_path, zip_file)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
            print(f"{zip_file} 압축 해제 완료!")
        
        os.remove(zip_path)
        print(f"{zip_file} 파일 삭제 완료!")
    
if __name__ == '__main__':
    download_data()
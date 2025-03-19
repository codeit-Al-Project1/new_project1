# Project1
# Data DownLoad Requirements

Kaggle 인증을 위해 kaggle.json을 kaggle 사이트에서 내려받아 인증할 필요성이 있다. 기본적으로 kaggle.json을 두는 위치는 C:\Users\<User_name>\.kaggle\kaggle.json 이다.

혹은 git bash로 환경 변수 설정을 두어 설정할수 있는데.

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

커멘드는 위를 참조하여 kaggle인증을 진행하면 된다.

인증 확인은 아래의 커멘드를 이용해 케글의 데이터셋 리스트를 불러오는 것으로 가능하다.

kaggle datasets list

또한, train_annotations을 최신화 시키기위해 google drive에서 데이터를 가져오게 되는데. 이떄 필요한 라이브러리 설치를 위해 아래의 커멘드를 실행할 필요가 있다.

pip install gdown


# Pull 활용 시 Rebase 문제

git config --global --get pull.rebase

위의 커멘드를 터미널에 입력하게 되면, 현재 git 설정에서 자동 rebase가 True인지 False인지를 보여 준다. 자동으로 덮어 씌워져 작업 데이터의 손실을 줄이고 변화를 확인하기 위해 False로 설정해야 한다.

git config --global pull.rebase false
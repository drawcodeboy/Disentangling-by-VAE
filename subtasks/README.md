### 1. data_ps
'''
python subtasks/data_ps/exec.py
'''
* 데이터셋 read & processing example
* .h5는 처음이라 핸들링 예시 필요
### 2. data_save
'''
python subtasks/data_save/exec.py
'''
* .h5는 데이터 로더 상에서 배치 단위로 한꺼번에 가져오려니 오랜 시간이 소요됨
* 이에 따라 .h5 파일 내 모든 이미지와 label들을 한 번에 가져오는 것이 필요함.
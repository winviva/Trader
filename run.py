import os

process_read = os.popen("ps -ef | grep try_my_luck.py | grep -v 'grep'").readlines()
# ps -ef 명령어를 이용해서 현재 프로세스를 출력한 후, 그 중 run24h.py 문자열이 포함된 줄만 모은다.
# grep 명령어 자체도 프로세스에 나타나므로 grep -v를 이용해서 제외한다.
check_process = str(process_read)
# 문자열로 변환한다.
text_location=check_process.find("try_my_luck.py")
# run24h.py가 몇번째 문자열인지 찾아낸다. 만약 문자열이 없으면, 즉 프로세스가 존재하지 않을 경우에는 -1을 반환한다.
if ( text_location == -1 ):
    print("Process not found!")
    os.system("python try_my_luck.py &")
    # 해당 프로그램을 다시 실행한다. 백그라운드에서 실행할 경우 &기호를 붙인다.
    print("Program restarted!")
else:
    print("Process exists. Location is",text_location)
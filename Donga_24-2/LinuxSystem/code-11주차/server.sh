#!/bin/bash

# 가상환경 경로 설정
VIRTUAL_ENV="/home/pjk_241110/flask/venv"               # 본인 프로젝트의 가상환경 경로로 수정하세요

# Flask 애플리케이션 실행 경로와 포트 설정
FLASK_APP="flask run --host=0.0.0.0 --port=15002"       # 본인 포트로 수정하세요
LOG_FILE="flask.log"
PID_FILE="flask.pid"

# 명령어에 따른 실행
case "$1" in
    start)
        # 가상환경 활성화
        if [ -d "$VIRTUAL_ENV" ]; then
            source "$VIRTUAL_ENV/bin/activate"
        else
            echo "가상환경을 찾을 수 없습니다: $VIRTUAL_ENV"
            exit 1
        fi

        # 서버가 이미 실행 중인지 확인
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "Flask 서버가 이미 실행 중입니다."
            exit 1
        fi

        echo "Flask 서버 시작 중..."
        nohup $FLASK_APP > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "Flask 서버가 시작되었습니다. (PID: $(cat $PID_FILE))"
        ;;

    stop)
        # 서버가 실행 중인지 확인하고 종료
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "Flask 서버 종료 중... (PID: $(cat $PID_FILE))"
            kill $(cat "$PID_FILE")
            rm "$PID_FILE"
            echo "Flask 서버가 종료되었습니다."
        else
            echo "실행 중인 Flask 서버를 찾을 수 없습니다."
        fi
        ;;

    *)
        echo "사용법: $0 {start|stop}"
        exit 1
        ;;
esac

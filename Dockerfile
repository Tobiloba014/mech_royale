FROM python:3.13-slim
WORKDIR /bot
COPY bot.py mechfight.py weights.json ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]

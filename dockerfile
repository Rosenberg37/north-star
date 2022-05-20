FROM python
RUN mkdir /app
WORKDIR /app
COPY . .
RUN pip install -U pytest

import pandas as pd
import json
from DB import MyPostgreSQL

from dotenv import load_dotenv
import os
from openai import OpenAI

class NO2SQL():

    def __init__(self) -> None :
        self.data = None
        self.db = None
        self.llm = None

    def load_data(self, path: str) -> None :
        # 데이터는 merged.csv + 각 query 마다 prompt 정보 열도 추가해야하나... prompt를 미리 다 세팅할 수 있나
        self.data = pd.read_csv(path)

    def connect_DB(self) -> None :
        self.db = MyPostgreSQL()
        self.db.login()
    
    # 초기 설정
    def create_all_tables(self) -> None : 

        self.connect_DB()
        
        CREATE_queries = self.data['CREATE'].unique()

        for query in CREATE_queries:
            self.db.execute_query(query)
        
    # 초기 설정
    def insert_all_tables(self) -> None : 

        self.connect_DB()

        INSERT_queries = self.data['INSERT'].unique()

        for query in INSERT_queries:
            self.db.execute_query(query)
    

    def get_answer_qSQL(self, qNL: str) -> str:
        qSQL = self.data[self.data['question'] == qNL]['query']

        return qSQL


    # 안써봐서 모름
    def connect_GPT_API(self) -> None:
        # 미리 .env file을 생성해서 OPENAI_API_KEY = sk-xxxx 형식으로 저장해둡니다.
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY is None:
            raise ValueError("API key not found in .env file")
        
        self.llm = OpenAI(api_key = OPENAI_API_KEY)
        pass

    def qNL_to_qSQL(self, qNL: str, schema: str = None, prompt: str = None) -> str:
        self.connect_GPT_API()
        # 'system': gpt-4에게 어떤 task를 수행할 지 지시합니다.
        # 'user': qNL, schema 등 입력할 내용을 넣어줍니다.
        
        system_message = "Translate the following natural language question into a SQL query compatible with PostgreSQL."
        if schema:
            system_message += f" The schema of database is as following: {schema}"
  
        response = self.llm.chat.completions.create(
        model= "gpt-4",
        messages= [
                {"role": "system", "content": system_message},
                {"role": "user", "content": qNL},
            ],
        temperature= 0
        )
        
        # 반환 받은 response에서 쿼리만 따로 추출합니다.
        qSQL = response.choices[0].message.content.replace('\n', '')        
        return qSQL
        
        # 예) qNL = 'How many departments are led by heads who are not mentioned?'
        #    qSQL = 'SELECT COUNT(*) FROM departments WHERE head_id NOT IN (SELECT id FROM heads)'
        
        # 사용한 token 수도 확인할 수 있습니다!
        # response.usage = CompletionUsage(completion_tokens=18, prompt_tokens=34, total_tokens=52)

    # 한 question에 대한 NO2SQL 결과
    def execute_NL2SQL(self, qNL: str, schema: str = None, prompt: str  =None) -> tuple:
        
        true_qSQL = self.get_answer_qSQL(qNL)
        true_qSQL_result = self.db.execute(true_qSQL)
        
        self.send_qNL_to_GPT(qNL)

        qSQL = self.receive_qSQL_from_GPT()
        qSQL_result = self.db.execute(qSQL)

        result = (true_qSQL, true_qSQL_result, qSQL, qSQL_result)

        return result


    def run_NO2SQLs(self) -> pd.DataFrame:

        ## 전체 데이터에 대해서 execute_NL2SQL 반복해서 결과를 dataframe으로 저장
        pass


if __name__ == "__main__":

    task = NO2SQL()
    task.connect_DB()
    qry = "SELECT datname FROM pg_database;"
    print(task.db.excute_query(qry))


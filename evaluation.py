import time
import json
import pandas as pd

from openai import OpenAI
from DB import MyPostgreSQL
from datetime import datetime

class NO2SQL():

    def __init__(self, data: pd.DataFrame, ddl: pd.DataFrame) -> None :
        """
        self.spider : qNL, qSQL, schema, prompt가 있는 데이터셋
        self.ddl : 각 db_id 에 해당하는 create, insert문이 있는 데이터셋
        self.db : DB.py를 통해 local postresql로 연결
        self.llm : chatgpt API
        """
        self.spider = data
        self.ddl = ddl
        
        self.db = MyPostgreSQL()
        self.db.login()

        self.llm = None

    def connect_GPT_API(self) -> None:
        """
        API_key를 가져와서 OpenAI API를 self.llm에 할당
        """
        
        with open("info.json", 'r') as file:
            info = json.load(file)
            OPENAI_API_KEY = info['gpt_info']['API_key']
        
        if OPENAI_API_KEY is None:
            raise ValueError("API key not found in .env file")
        
        self.llm = OpenAI(api_key = OPENAI_API_KEY)

    def qNL_to_qSQL(self, model: str, db_id: str, qNL: str, schema: bool = False, prompt: bool = False, token: bool = False) -> str:
        """
        model : 사용할 model - gpt-3.5-turbo, gpt-4
        db_id : schema불러올 self.ddl 조회용
        qNL : gpt한테 보낼 자연어 질문
        schema : True면 CREATE문을 함께 보내줌
        prompt : True면 PROMPT를 함께 보내줌
        token : API 사용량 출력
        """
        self.connect_GPT_API()

        # 'system': gpt-4에게 어떤 task를 수행할 지 지시합니다.
        # 'user': qNL, schema 등 입력할 내용을 넣어줍니다.
        system_message = """
                            Translate the following natural language question into a SQL query compatible with PostgreSQL.
                         """
        
        if schema:
            schema_info = ""
            current_schema = self.ddl[self.ddl['db_id'] == db_id].copy()

            for row in current_schema.iterrows():
                schema_info += f"{row[1]['CREATE']}\n"
            
            system_message += f" The schema of database is as following: \n{schema_info}\n"
            print(system_message)

        ## prompt 어떤식으로 반영할지 고민해야 함 -> SPIDER_SELECTED.csv에 한 column 만들어서 추가하는 방식으로 구현
        if prompt:
            pass
        print(system_message)
        response = self.llm.chat.completions.create(
                        model= model,
                        messages= [
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": qNL},
                                  ],
                        temperature= 0
                    )
        
        # 반환 받은 response에서 쿼리만 따로 추출합니다.
            # 예) qNL = 'How many departments are led by heads who are not mentioned?'
            #    qSQL = 'SELECT COUNT(*) FROM departments WHERE head_id NOT IN (SELECT id FROM heads)'
        qSQL = response.choices[0].message.content.replace('\n', ' ')        
        
        if token:
            print(response.usage)

        return qSQL

    def create_schema(self, db_id: str, verbose: bool = False):
        """
        db_id : 어떤 테이블을 생성하고 데이터를 삽입할지 알기 위해 필요
        verbose : 진행상황 출력여부
        """
        # db_id에 해당하는 table 생성 및 데이터 삽입
        # 테이블 이름이 동일한 경우가 많아서 단순하게 매번 새롭게 table 삭제 -> table 생성 -> 데이터 삽입 과정 거침

        # 기존 테이블 모두 제거
        self.db.drop_all_tables(verbose=verbose)

        # db_id에 해당하는 table 추출
        current_schema = self.ddl[self.ddl['db_id'] == db_id].copy()
        if len(current_schema) == 0:
            print("db_id does not exist")
        
        try:
            for row in current_schema.iterrows():
                create_qry, insert_qry = row[1]['CREATE'], row[1]['INSERT']
                self.db.execute_query(create_qry)
                self.db.execute_query(insert_qry)
            
            if verbose:
                print("Created schema successfully")
        
        except Exception as e:
            print(f"Error: {str(e)}")


    def get_SQL_answer(self, qNL: str) -> str:
        """
        qNL이 주어지면 self.spider에서 정답쿼리 return
        """

        return self.spider[self.spider['question'] == qNL].loc[:, 'query'].values[0]


    # 한 question에 대한 NO2SQL 결과
    def execute_NL2SQL(self, model: str, db_id: str, qNL: str, schema: bool = False, prompt: bool = False, token: bool = False) -> tuple:
        """
        한 qNL이 주어졌을 때, gpt로부터 얻은 답 + 실제 정답 + 각 쿼리로 얻어진 결과를 return
        """

        # 필요한 테이블 및 데이터 추가
        self.create_schema(db_id, verbose=False)

        # spider 데이터셋에 있는 정답 쿼리       
        true_SQL = self.get_SQL_answer(qNL)
        print(f"true_SQL : {true_SQL}")
        
        # gpt로부터 얻은 쿼리
        qSQL = self.qNL_to_qSQL(model, db_id, qNL, schema, prompt, token)
        print(f"qSQL : {qSQL}")

        # 정답쿼리로 출력한 결과
        true_SQL_result = self.db.select_query(true_SQL)
        print(f"true_SQL_result : \n{true_SQL_result}")
        
        # qSQL로 출력한 쿼리
        qSQL_result = self.db.select_query(qSQL)
        print(f"qSQL_result : \n{qSQL_result}\n")

        # 쿼리 결과 dataframe은 array로 바꿔서 저장
        result = (true_SQL, qSQL, true_SQL_result.values, qSQL_result.values)

        return result
    

    def run_NO2SQLs(self, model: str, schema: bool = False, prompt: bool = False, token: bool = False) -> pd.DataFrame:

        # 전체 데이터에 대해서 execute_NL2SQL 반복해서 결과를 dataframe으로 저장

        total_result = []
        for row in self.spider.iterrows():
            db_id, qNL = row[1]['db_id'], row[1]['question']
            print(f"db_id : {db_id},  \nqNL : {qNL}")
            
            result = self.execute_NL2SQL(model, db_id, qNL, schema, prompt, token)
            total_result.append(result)

        total_result = pd.DataFrame(total_result, columns=['true_SQL', 'qSQL', 'true_SQL_result', 'qSQL_result'])
        total_result.to_csv(f"result/{datetime.now().strftime('%m%d %H:%M')}_schema:{schema}_prompt:{prompt}_model:{model}.csv")

        return total_result

    def working_test(self, db_id, qNL):
        """
        SPIDER_SELECTED.csv query, CREATE, INSERT문 잘 동작하는지 확인하기 위한 함수
        """
        
        self.create_schema(db_id,verbose=False)

        # spider 데이터셋에 있는 정답 쿼리       
        true_SQL = self.get_SQL_answer(qNL)
        print(f"true_SQL : {true_SQL}")
        
        true_SQL_result = self.db.select_query(true_SQL)
        print(f"true_SQL_result : \n{true_SQL_result}\n")

    def run_NL2SQLs_test(self, model):
        """
        SPIDER_SELECTED.csv query, CREATE, INSERT문 잘 동작하는지 확인하기 위한 함수
        """
        for row in self.spider.iterrows():
            db_id, qNL = row[1]['db_id'], row[1]['question']
            print(f"db_id : {db_id},  \nqNL : {qNL}")
            self.working_test(db_id, qNL)


if __name__ == "__main__":

    # 전체 실행
    spider = pd.read_csv("rawdata/SPIDER_SELECTED.csv", encoding='cp949')
    ddl = pd.read_csv("rawdata/DDL_SELECTED.csv")

    task = NO2SQL(spider, ddl)
    
    start = time.time()
    result = task.run_NO2SQLs(model='gpt-3.5-turbo', schema=True, token=True) # gpt-3.5-turbo, gpt-4
    end = time.time()

    print(result)
    print(f"execution time : {end - start:.5f} sec")



        


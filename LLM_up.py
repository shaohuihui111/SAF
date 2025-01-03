import openai
import os
import requests
import re
from abc import abstractmethod
# from KnowledgeGraph import KnowledgeGraph,Relation


def Str2MessageList(s:str)->list:
    lines=[line.strip() for line in s.strip().split("\n")]
    messageList = [{"role":line.split(':')[0],"content":line.split(':')[1]} for line in lines]
    return messageList


def Message2Str(messageList:list)->str:
    try:
        lines = [f"{message['role']}:{message['content']}" for message in messageList]
        s = '\n'.join(lines)
        return s
    except KeyError as e:
        print(f"{e}\nIllegal message form.")


class LLM:
    def __init__(
        self, 
        baseUrl: str, 
        apiKey: str,
        model: str = "llama3:70b", 
        maxTokens: int = 2048, 
        temperature: float = 0.0,  # 确保参数命名正确
    ):
        self.baseUrl = baseUrl
        self.apiKey = apiKey
        self.model = model
        self.maxTokens = maxTokens
        self.temperature = temperature  # 修正拼写
        openai.api_key = self.apiKey
        openai.base_url = self.baseUrl

        
        
    @abstractmethod
    def Complete(self,prompt:str)->str:
        pass
    
    @abstractmethod
    def Chat(self,message:list)->list:
        pass
    
    @abstractmethod
    def SegmentSentence(self, sentence:str)->list:
        pass

class CompletionLLM(LLM):
    def __init__(
        self, 
        baseUrl: str, 
        apiKey: str,
        model: str = "llama3:70b", 
        maxTokens: int = 2048, 
        temperature: float = 0.0,  # 子类接受相同的参数
    ):
        # 明确传递参数给父类
        super().__init__(
            baseUrl=baseUrl,
            apiKey=apiKey,
            model=model,
            maxTokens=maxTokens,
            temperature=temperature,  # 参数传递给父类
        )
        self.tries = 5

    
    
    def Complete(self,prompt)->str:
        for _ in range(self.tries):
            try:
                res=openai.completions.create(
                    model=self.model,
                    max_tokens=self.maxTokens,
                    temperature=self.temperature,
                    prompt=prompt
                )
                return res.choices[0].text
            except Exception as e:
                print(f"Error occurred: {e}")  # 打印具体错误信息
                raise  # 或者终止继续尝试except:

        
        #raise openai.OpenAIError("Failed to get LLMs' answer.")
        
        
    def Chat(self,message:list)->list:
        for _ in range(self.tries):
            try:
                res=openai.chat.completions.create(
                    model=self.model,
                    max_tokens=self.maxTokens,
                    temperature=self.temperature,
                    message=message
                )
                return res.choices[0].text
            except Exception as e:
                print(f"Error occurred: {e}")  # 打印具体错误信息
                raise  # 或者终止继续尝试except:
        
        #raise openai.OpenAIError("Failed to get LLMs' answer.")
        
    
    
    def SegmentSentence(self,sentence:str,debug:bool=False)->list:
        """将长句分割为若干个短句并且抽取生成的短句的实体，通过指定prompt来确定分割数量

        Args:
            sentence (str): 待分割的长句
            debug (bool): 是否输出调试信息

        Returns:
            list[dict[substence:"",entitySet:[str]]]: 返回子句:实体集合表
        """        
        prompt=self.sentenceSegmentationPrompt.replace("<<<<CLAIM>>>>", sentence).replace("<<<<ENTITY_SET>>>>",str(KnowledgeGraph.ExtractEntities(sentence)))
        reply = self.Complete(prompt)
        if debug:
            print(f"User:{prompt}")
            print(f"LLM:{reply}")
        lines = reply.split("\n")
        subsentenceWithEntitySet = []
        for line in lines:
            if re.match(r"[0-9]+\. .*, Entity set: \[.*\]", line) is not None:
                ponitIndex=line.find('.')
                line=line[ponitIndex+2:]# 去掉数字标号
                entitySetIndex=line.find(", Entity set: ")
                entitySetText=line[entitySetIndex+14:].strip()
                subsentence=line[:entitySetIndex].strip()
                entitySet=[entity.strip()[1:-1].lower() for entity in entitySetText[1:-1].split(',')]
                # 移除空字符串
                while True:
                    try:
                        entitySet.remove('')
                    except:
                        break
                subsentenceWithEntitySet.append({"subsentence":subsentence,"entitySet":entitySet})
        return subsentenceWithEntitySet
    
    
    def TopKRelations(self,sentence:str,rels:list,k=2):
        """从rels中选取k个和短句中蕴含的关系词最接近的词

        Args:
            sentence (str): 短句
            rels (list): 关系词表
            k (int, optional): topk. Defaults to 2.
        """
        prompt=self.relationRetrievalPrompt.replace("<<<<TOP_K>>>>",str(k)).replace("<<<<SENTENCE>>>>",sentence).replace("<<<<RELATION_SET>>>>",str(rels))
        answer = self.Complete(prompt)
        # print(answer)
        try:
            topkRelsText=re.findall(r"\[.+?\]",answer)[0]
            # topkRelsText=re.findall(r"\[.+?\]",answer)[0]
            # topkRelsText=re.findall(r"\[.+?\]",answer)[0]
        except IndexError as e:
            print(f"LLM receive 「{prompt}」")
            print(f"LLM said 「{answer}」, didn't return the answer.")
        except:
            print(f"{answer}")
        topkRels = [rel.strip() for rel in topkRelsText.strip().replace("'","")[1:-1].split(",")]
        return topkRels
    
    
    def AnswerWithEvidence(self, claim:str):
        subsentenceWithEntitySet = self.SegmentSentence(claim)
        

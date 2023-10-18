
import logging
# 1、创建一个logger
import os
class Logger:
    #日志级别关系映射
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    def __init__(self, name, save_file):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 2、创建一个handler，用于写入日志文件
        fh = logging.FileHandler(save_file)
        fh.setLevel(logging.DEBUG)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 3、定义handler的输出格式（formatter）
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 4、给handler添加formatter
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 5、给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    def get_logger(self):
        return self.logger


class EvalLogger:
    def __init__(self,filePath,ioModel='a'):
        # if not os.path.exists(filePath):
        #   print(123)
        #   fp = open(filePath,'w',encoding='utf8')
        #   fp.close()
        self.filePath = filePath
        self.ioModel = ioModel
        self.f = open(self.filePath,self.ioModel,encoding='utf8')

    def __del__(self):
        self.f.close()

    def write(self,strdata):
        self.f.write(str(strdata))

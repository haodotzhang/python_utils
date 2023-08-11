import os, csv, json
import pandas as pd

class FileHelper():
    def __init__(self) -> None:
        pass


class PathHelper():
    def __init__(self) -> None:
        pass

    def get_path_v1(self):
        # 获取当前目录
        p1_1 = os.getcwd()
        p1_2 = os.path.abspath('.')
        p1_3 = os.path.abspath(os.path.dirname(__file__))
        # 获取上一级目录
        p2_1 = os.path.abspath('..')
        p2_2 = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        p2_3 = os.path.abspath(os.path.dirname(os.getcwd()))
        p2_4 = os.path.abspath(os.path.join(os.getcwd(), ".."))
        # 获取上上一级目录
        p3_1 = os.path.abspath('../..')
        p3_2 = os.path.abspath(os.path.join(os.getcwd(), "../.."))


class CSVHelper():
    '''
    - [1].https://zhuanlan.zhihu.com/p/506858056 
          https://zhuanlan.zhihu.com/p/505002385  
    '''
    def __init__(self) -> None:
        pass

    # 读取到 list [[], ]
    def csv_reader_v1(self, path_input):
        with open(path_input, "r", encoding="utf8") as fp:
            csv_reader = csv.reader(fp)
            # skip the first row
            # next(csv_reader) # 跳过第一行的标题行
            # show the data
            ret_list = []
            for line_no, line in enumerate(csv_reader, 1):
                if line_no == 1:
                    header_list = line 
                else:
                    ret_list.append(line) # 逐个取出每一行
        
        return ret_list, header_list
    
    # 读取到 dict [{}, ]
    def csv_reader_v2(self, path_input, header_list=None):
        with open(path_input, "r", encoding="utf8") as fp:
            if header_list is not None:
                csv_reader = csv.DictReader(fp, header_list) # header_list 支持自定义覆盖原始文件的标题行
            else:
                csv_reader = csv.DictReader(fp) # 使用文件默认的标题行
            # skip the first row
            # next(csv_reader)
            # show the data
            ret_list = []
            for line in csv_reader: # 注意dict方式 第一行就是数据 不存在标题行
                ret_list.append(line) # 逐个取出每一行

            if header_list is None:
                header_list = list(line.keys())
        
        return ret_list, header_list

    # 写入list
    def csv_writer_v1_1(self, path_output, header_list, data_list):
        with open(path_output, "a+" ,encoding='utf8', newline='') as fp:
            writer = csv.writer(fp)
            # write the header
            writer.writerow(header_list)
            # write the data
            for data in data_list: # 多次写入
                writer.writerow(data) # 每次写一行 list

    def csv_writer_v1_2(self, path_output, header_list, data_list):
        with open(path_output, "a+" ,encoding='utf8', newline='') as fp:
            writer = csv.writer(fp)
            # write the header
            writer.writerow(header_list)
            # write the data
            writer.writerows(data_list) # 直接一次写多行 [[],]

    # 写入dict
    def csv_writer_v2_1(self, path_output, header_list, data_list):
        with open(path_output, "a+" ,encoding='utf8', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=header_list)
            writer.writeheader()
            for data in data_list: # 多次写入
                writer.writerow(data) # 每次写一行 dict

    def csv_writer_v2_2(self, path_output, header_list, data_list):
        with open(path_output, "a+" ,encoding='utf8', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=header_list)
            writer.writeheader()
            writer.writerows(data_list) # 直接一次写多行 [{},]


class JsonHelper():
    def __init__(self) -> None:
        pass

    def json_read_v1(self):
        pass 

    def json_write_v1(self):
        pass


class TxtHelper():
    pass 


class XmlHelper():
    pass


class ExcelHelper():
    pass 


def test_CSVHelper_v1():
    root_path = os.getcwd()
    print("root_path: ", root_path)
    csv_helper = CSVHelper()

    def test_csv_writer_v1_1():
        path_output = os.path.join(root_path, "data_dir/test_csv_writer_v1_1.csv")
        header_list = ['id', 'stu_id', 'course_name', 'course_score']
        data_list = [
            [1, 1, 'English', 100],
            [2, 1, 'Math', 95],
            [3, 2, 'English', 96]
        ]
        csv_helper.csv_writer_v1_1(path_output, header_list, data_list)
    test_csv_writer_v1_1()

    def test_csv_writer_v1_2():
        path_output = os.path.join(root_path, "data_dir/test_csv_writer_v1_2.csv")
        header_list = ['id', 'stu_id', 'course_name', 'course_score']
        data_list = [
            [1, 1, 'English', 100],
            [2, 1, 'Math', 95],
            [3, 2, 'English', 96]
        ]
        csv_helper.csv_writer_v1_2(path_output, header_list, data_list)
    test_csv_writer_v1_2()

    def test_csv_writer_v2_1():
        path_output = os.path.join(root_path, "data_dir/test_csv_writer_v2_1.csv")
        header_list = ['id', 'stu_id', 'course_name', 'course_score']
        data_list = [
            {'id': 1,
            'stu_id': 1,
            'course_name': 'English',
            'course_score': 100},

            {'id': 2,
            'stu_id': 1,
            'course_name': 'Math',
            'course_score': 95},

            {'id': 3,
            'stu_id': 2,
            'course_name': 'English',
            'course_score': 96}
        ]
        csv_helper.csv_writer_v2_1(path_output, header_list, data_list)
    test_csv_writer_v2_1()

    def test_csv_writer_v2_2():
        path_output = os.path.join(root_path, "data_dir/test_csv_writer_v2_2.csv")
        header_list = ['id', 'stu_id', 'course_name', 'course_score']
        data_list = [
            {'id': 1,
            'stu_id': 1,
            'course_name': 'English',
            'course_score': 100},

            {'id': 2,
            'stu_id': 1,
            'course_name': 'Math',
            'course_score': 95},

            {'id': 3,
            'stu_id': 2,
            'course_name': 'English',
            'course_score': 96}
        ]
        csv_helper.csv_writer_v2_2(path_output, header_list, data_list)
    test_csv_writer_v2_2()

def test_CSVHelper_v2():
    root_path = os.getcwd()
    print("root_path: ", root_path)
    csv_helper = CSVHelper()

    def test_csv_reader_v1():
        path_input = os.path.join(root_path, "data_dir/test_csv_writer_v2_2.csv")
        ret_list, header_list  = csv_helper.csv_reader_v1(path_input)
        print("header_list: ", header_list)
        print("test_csv_reader_v1: ", ret_list)
        print("=========")
    test_csv_reader_v1()
    
    def test_csv_reader_v2_1():
        path_input = os.path.join(root_path, "data_dir/test_csv_writer_v2_2.csv")
        ret_list, header_list  = csv_helper.csv_reader_v2(path_input)
        print("header_list: ", header_list)
        print("test_csv_reader_v2_1: ", ret_list)
        print("=========")
    test_csv_reader_v2_1()

    def test_csv_reader_v2_2():
        path_input = os.path.join(root_path, "data_dir/test_csv_writer_v2_2.csv")
        header_list = ['标识符', '学生编号', '课程', '成绩']
        ret_list, header_list  = csv_helper.csv_reader_v2(path_input, header_list)
        print("header_list: ", header_list)
        print("test_csv_reader_v2_2: ", ret_list)
        print("=========")
    test_csv_reader_v2_2()



if __name__ == "__main__":
    test_CSVHelper_v1()
    # test_CSVHelper_v2()

    pass
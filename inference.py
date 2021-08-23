import fitz
import json
import requests
from tqdm import tqdm
from datetime import datetime
from fuzzywuzzy import fuzz
from dateutil import parser as pr
from utils import *
from CFS import *
# from elasticsearch import Elasticsearch, helpers

# es = Elasticsearch(hosts=["http://localhost:9200"], timeout=240, max_retries=2, retry_on_timeout=True)

time_check = datetime.datetime.strptime('Jul 31 2021', '%b %d %Y')
reader = ocr_engine.Reader()
provider = os.getenv('PROVIDER', 'CUDAExecutionProvider')
lsq_model = onnxruntime.InferenceSession("models/yolov5/lsq.onnx", providers=[provider])
date_model = onnxruntime.InferenceSession("models/yolov5/date.onnx", providers=[provider])


def getInfoCFS(json_data, reader, lsq_model, date_model, time_check = datetime.datetime.now()):
    pdf_paths = [".tmp/cfs.pdf"]  
    cfs = CFS()
    FLAGS = [0,0,0,0,0,0]
    COMMENTS = [0,0,0,0,0,0,0]
    attachmentFileList = []
    ERR = []
    no_eqt = []
    no_code = []
    if pdf_paths:
        for doc_number, pdf_path in enumerate(pdf_paths):
            flags = [0,0,0,0,0,0]
            
            if 0 in flags:
                try:
                    text = cfs.tool_reader(pdf_path)
                except:
                    if doc_number<= len(pdf_path):
                        continue
                    return None, None, None, None
                
                comments = []
                if "PHIẾU TIẾP NHẬN" in text['page_1']:
                    output = cfs.get_ptn_info(text)
                    final_result, doc_type, err, flags, COMMENTS, no_code, no_eqt = cfs.final_result(query=json_data,vertical_result=output, no_eqt=no_eqt, no_code=no_code, ptn=True,flags=flags, COMMENTS=COMMENTS, doc_number=doc_number+1, time_check=time_check)
                    ERR.extend(err)
                    comments.extend(final_result)
                    print('ptn_flags: ', flags)
                else:
                    vertical_results, horizontal_results, lsq_detect, images = cfs.deeplearning_reader(pdf_path, reader=reader, lsq_model=lsq_model, date_model=date_model, check_annot=True)                
                    
                    # try:
                    # x = push_result(vertical_results,horizontal_results)
                    
                    final_result, doc_type, err, flags, COMMENTS, no_eqt, no_code = cfs.final_result(query=json_data, file_pdf=pdf_path, vertical_results=vertical_results, horizontal_results=horizontal_results, no_eqt=no_eqt, no_code=no_code, check_annot=True, flags=flags, COMMENTS=COMMENTS, doc_number=doc_number+1, time_check=time_check)
                    
                    if lsq_detect[0]:
                        final_result.append({
                            "commentContent" : "Có dấu lãnh sự quán | trang {"+f"{lsq_detect[1]}"+"}",
                            "commentStatus" : "OK"
                        })
                        flags[5] = 1
                        if COMMENTS[5]:
                            COMMENTS[5] = COMMENTS[5] + f', tài liệu số {doc_number+1} | trang {lsq_detect[1]}'
                        else:
                            COMMENTS[5] = f'Có dấu lãnh sự quán tài liệu số {doc_number+1} | trang {lsq_detect[1]}'
                    comments.extend(final_result)
                    if 0 in flags:
                        vertical_results, horizontal_results, _ = cfs.deeplearning_reader(pdf_path, reader=reader, lsq_model=lsq_model, date_model=date_model, check_annot=False, images=images)
                        
                        final_result, doc_type, err, flags, COMMENTS, no_eqt, no_code = cfs.final_result(query=json_data, file_pdf=pdf_path, vertical_results=vertical_results, horizontal_results=horizontal_results, no_eqt=no_eqt, no_code=no_code, check_annot=False, flags=flags, doc_number=doc_number+1, COMMENTS=COMMENTS, time_check=time_check)
                        
                        comments.extend(final_result)

            if not flags[5]:
                comments.append({
                                    "commentContent" : f"Không có dấu lãnh sự quán",
                                    "commentStatus" : "NOK"
                                })
                ERR.append(7)

            attachmentFileList.append({'fileCommentList': comments})
            
            FLAGS = list(map(sum, zip(FLAGS,flags)))

    else:
        comments = {
                    "commentContent" : f"Không có giấy CFS hoặc không tải được CFS",
                    "commentStatus" : "NOK"
                    }   
    
    RESULTS = []
    if COMMENTS[-1]:
        RESULTS = {
            'attachmentComment': COMMENTS[-1],
            'attachmentStatus': 'OK'
            }
    elif 0 not in FLAGS:
        RESULTS = {'attachmentComment': 'Giấy lưu hành tự do phù hợp với đơn đăng ký',
                    'attachmentStatus': 'OK'}
    else:
        if COMMENTS[0]:
            pass
            # RESULTS.append({
            # 'attachmentComment': COMMENTS[0],
            # 'attachmentStatus': 'OK'
            # })
        else:
            RESULTS.append({
            'attachmentComment': 'Tên công ty sản xuất không phù hợp với đơn đăng ký',
            'attachmentStatus': 'NOK'
            })
        if COMMENTS[1]:
            pass
            # RESULTS.append({
            # 'attachmentComment': COMMENTS[1],
            # 'attachmentStatus': 'OK'
            # })
        else:
            RESULTS.append({
            'attachmentComment': 'Tên công ty sở hữu không phù hợp với đơn đăng ký',
            'attachmentStatus': 'NOK'
            })
        if COMMENTS[2]:
            pass
            # RESULTS.append({
            # 'attachmentComment': COMMENTS[2],
            # 'attachmentStatus': 'OK'
            # })
        else:
            RESULTS.append({
            'attachmentComment': 'Danh sách chủng loại TTBYT ko đúng',
            'attachmentStatus': 'NOK'
            })
        if COMMENTS[3]:
            pass
            # RESULTS.append({
            # 'attachmentComment': COMMENTS[3],
            # 'attachmentStatus': 'OK'
            # })
        else:
            RESULTS.append({
            'attachmentComment': 'Danh sách mã TTBYT ko đúng',
            'attachmentStatus': 'NOK'
            })
        if COMMENTS[4]:
            pass
            # RESULTS.append({
            # 'attachmentComment': 'Thời gian còn hiệu lực',
            # 'attachmentStatus': 'OK'
            # })
        else:
            RESULTS.append({
            'attachmentComment': 'Thời gian hết hiệu lực',
            'attachmentStatus': 'NOK'
            })
        if COMMENTS[5]:
            pass
            # RESULTS.append({
            # 'attachmentComment': COMMENTS[5],
            # 'attachmentStatus': 'OK'
            # })
        else:
            RESULTS.append({
            'attachmentComment': 'Không có dấu lãnh sự quán',
            'attachmentStatus': 'NOK'
            })
        
    return attachmentFileList, RESULTS, FLAGS, ERR

if __name__ == '__main__':
    
    json_folder = 'input_dmec'
    checkpoint = []
    check_valid = []
    check_sig = []
    
    files = ['33204_000.00.04.G18-210427-0018.json']
    # for file in os.listdir(json_folder):
    for file in files:
        file_path = file
        json_file = os.path.join(json_folder,file_path)
        attachmentFileList, RESULTS, FLAGS, err = getInfoCFS(json_file,file_path, check_valid, check_sig)
        result = {
                    'attachmentFileList': attachmentFileList,
                    'result': RESULTS
                }
        print(result)
        if not RESULTS:
            if not os.path.exists('file_fail.json'):
                with open('file_fail.json','w') as jsonfile:
                    json.dump([], file)
            file_err = json.load(open('file_fail.json','r'))
            with open('file_fail.json','w') as jsonfile:
                json.dump(file_err, jsonfile)
            continue
        with open(os.path.join('output',file_path),'w') as output:
            json.dump(result,output)

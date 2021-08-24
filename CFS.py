import fitz
from utils import *
import json
import numpy as np
import time 
import datetime
from dateutil import parser as pr
import datefinder
from fuzzywuzzy import fuzz
from ocr import ocr_engine


# from txtai.pipeline import Similarity

# from elasticsearch import Elasticsearch, helpers
# import easyocr
# reader = easyocr.Reader(['vi','en'], verbose=False, gpu = True)


# es = Elasticsearch(hosts=["http://localhost:9200"], timeout=240, max_retries=2, retry_on_timeout=True)

class CFS(object):

    def __init__(self):
        self.pdf_path = None
        self.is_local = True
        self.info = {}
        # self.reader = {
        #     'Tool' : self.tool_reader,
        #     'AI' : self.deeplearning_reader
        # }
        self.pdf_info = {}

    def tool_reader(self, pdf_path):
        pdf_info = {}
        doc = fitz.open(pdf_path)
        p = 0
        for page in doc:
            pdf_info[f'page_{p + 1}'] = []
            blocks = json.loads(page.getText('json'))['blocks']
            blocks = sort_block(blocks)
            for block in blocks:
                text = get_text(block)
                pdf_info[f'page_{p + 1}'].append(text)
            p += 1
        return pdf_info

    def deeplearning_reader(self, pdf_path, reader, lsq_model, date_model, check_annot=True, images=None):
        if not check_annot:
            
            vertical_results = []
            horizontal_results = []
            for p, img in enumerate(images):
                result, horizontal_result = reader.readtext(img, paragraph=False, cluster=True, page=p+1)
                if result:
                    vertical_results.extend(result)
                if horizontal_result:
                    horizontal_results.extend(horizontal_result)
                p+=1
                print('++'*30)
            return vertical_results, horizontal_results, images
        # doc = fitz.open(pdf_path)
        # p = 0
        pdf_info = {}
       
        p = 0
        t1 = time.time()
        images, annots, lsq_detect = pdfimage_process(pdf_path, lsq_model, date_model, check_annot=check_annot)
        print(time.time()-t1)
        # print(annots)
        # exit()
        vertical_results = []
        horizontal_results = []
        if check_annot:
            for p, annot in enumerate(annots):
                if annot is not None:
                    # print('=='*30)
                    # print(annot)
                    result, horizontal_result = reader.readtext(annot, paragraph=False, cluster=False, page=p+1)
                    if result:
                        vertical_results.extend(result)
                    if horizontal_result:
                        horizontal_results.extend(horizontal_result)
                    # pdf_info[f"page_{p + 1}"] = result
                p += 1
        

        return vertical_results, horizontal_results, lsq_detect, images

    def get_ptn_info(self, pdf_info):
        result = {
            "registrant" : {
                "name" : "",
                "address" : ""
            },
            "factory" : {
                "name" : "",
                "address" : "",
            },
            "equipment" : ""
        }

        for page in pdf_info.keys():
            flag = False
            for text in pdf_info[page]:
                print(text)
                print("----------------------")
                if "1. Tên cơ sơ công bố" in text:
                    result['registrant']['name'] = text.split(':')[-1].strip()
                elif "2. Địa chỉ" in text:
                    result['registrant']['address'] = text.split('(')[0].split(':')[-1]
                if '(Sản xuất tại:' in text:
                    result['factory']['name'] = text.split('(')[-1].strip().split(";")[0].split(":")[-1].strip()
                    result['factory']['address'] = text.split(";")[-1].replace(')', '')
                elif "5. Tên trang thiết bị" in text:
                    flag = True
                    continue
                if flag:
                    result['equipment'] = text
                    flag = False
        return result

     
    def get_info(self, info):
        if isinstance(info, str):
            info = json.load(open(info, 'w'))
        
        key_info = {
            "equipment" : info['equipment'],
            "equipmentOwner" : info['equipmentOwner']
        }
        if len(info['cfsForeign']) > 0 and len(info['cfsLocal']) == 0:
            self.is_local = False

        
        if self.is_local:
            urls = info['cfsLocal']['files']
            for url in urls:
                pdf_path = save_pdf(url['url'])
                try:
                    pdf_info = self.reader['Tool'](pdf_path)
                except:
                    pdf_info = self.reader['AI'](pdf_path)
                self.pdf_info[f"file_{url[id]}"] = pdf_info


        else:
            urls = info['cfsForeign']['files']
            for url in urls:
                pdf_path = save_pdf(url['url'])
                pdf_info = self.reader['AI'](pdf_path)
                self.pdf_path[f'file_{url[id]}'] = pdf_info

    def final_result(self, query, ptn=False, file_pdf=None, vertical_results=[], horizontal_results=[], no_eqt=[], no_code=[], check_annot=True, flags=[], COMMENTS=[], doc_number=None, time_check=datetime.datetime.now()):
        comments = []
        err = []

        factory_name = None
        factory_process = {'co.':'', 'ltd':'', 'inc':'', 'company':'', 'industr':'', 'việt nam':'', 'vietnam':'', 'trung quốc':'', 'công ty':'', 'tnhh':''}
        for info in query['equipment']['typeList']:
            if info['factory']['name']:
                factory_name = info['factory']['name'].split('(')[0].split('/')[0]
                factory_name = re.sub(' +', ' ',replace_all(factory_name.lower(), factory_process))
                break
        print('factory_name: ', factory_name)

        owner_process = {'co.':'', 'ltd':'', 'inc':'', 'company':'','industr':'', 'việt nam':'', 'vietnam':'', 'trung quốc':'', 'công ty':'', 'tnhh':''}
        owner_name = query["equipmentOwner"]['name'].split('(')[0].split('/')[0]
        owner_name = re.sub(' +', ' ',replace_all(owner_name.lower(), owner_process))
        print('owner_name: ', owner_name)

        eqt_name = []
        eqt_types = []
        for info in query['equipment']['typeList']:
            print('type: ',info['type'])
            # if info['name']:
            if info['type']:
                # eqt = info['name'].replace(';',',')
                eqt_type = [type.strip() for type in info['type'].replace(';',',').replace('\r', '').replace('\n',',').split(',')]
                eqt_types.extend(eqt_type)
                # if eqt_type:
                #     eqt_name.append([eqt,eqt_type])
        print('eqt_name: ', eqt_types)

        eqt_code = []
        for info in query['equipment']['typeList']:
            if info['code']:
                codes = [code for code in info['code'].replace(';',',').replace('\r', '').replace('\n',',').split(',')]
                eqt_code.extend(codes)
        print('eqt_code: ', eqt_code)

        punc_search = {':':'','/':' ','{':' ','}':' ','(':' ',')':' ','[':' ',']':' ','^':'','"':''}
        
        codes_log = None

        if ptn:
            if fuzz.token_set_ratio(vertical_results['factory']['name'], factory_name) > 60 or fuzz.token_set_ratio(factory_name, vertical_results['factory']['name']) > 60:
                comments.append(
                    {
                        "commentContent" : "Tên công ty sản xuất phù hợp với đơn đăng ký | trang 1",
                        "commentStatus" : "OK"
                    }
                )
                
                if not COMMENTS[-1]:
                    COMMENTS[-1] = 'Phiếu tiếp nhận phù hợp với đơn đăng ký'

            else:
                comments.append(
                    {
                        "commentContent" : "Tên công ty sản xuất không phù hợp với đơn đăng ký | trang 1",
                        "commentStatus" : "NOK"
                    }

                )
                err.append(1)
            doc_type = 1
            flags = [1,1,1,1,1,1]
            return comments, doc_type, err, flags, COMMENTS
        
        else:
            ####
            # phieutiepnhan = search('PHIẾU TIẾP NHẬN',3)
            # print('phieutiepnhan: ', phieutiepnhan)
            print('vertical_result: ', vertical_results)
            for ptn in vertical_results:
                if "phiếu tiếp nhận" in ptn.lower():
                    name_equipmentFactory = normal_search(vertical_results, horizontal_results, replace_all(factory_name, punc_search), score=60)
                    
                    if name_equipmentFactory[0]:
                        
                        comments.append({
                                "commentContent" : f"Tên công ty sản xuất phù hợp với đơn đăng ký | trang {name_equipmentFactory[1][-7:].split(' ')[-1]}",
                                "commentStatus" : "OK"
                            })
                        
                        if not COMMENTS[-1]:
                            COMMENTS[-1] = 'Phiếu tiếp nhận phù hợp với đơn đăng ký'

                    else:
                        comments.append({ 
                            "commentContent" : "Tên công ty sản xuất không phù hợp với đơn đăng ký",
                            "commentStatus" : "NOK"
                        })
                        err.append(1)
                    doc_type = 1
                    flags = [1,1,1,1,1,1]
                    return comments, doc_type, err, flags, COMMENTS

            if not flags[0]:
                if factory_name:
                    print('vertical_res: ', vertical_results)
                    name_equipmentFactory = normal_search(vertical_results, horizontal_results, replace_all(factory_name, punc_search), score=60)
                    # print('name_equipmentFactory: ', name_equipmentFactory)
                    if name_equipmentFactory[0]:
                        comments.append({
                                "commentContent" : f"Tên công ty sản xuất phù hợp với đơn đăng ký | trang {name_equipmentFactory[1][-7:].split(' ')[-1]}",
                                "commentStatus" : "OK"
                            })
                        flags[0] = 1

                        if COMMENTS[0]:
                            COMMENTS[0] = COMMENTS[0] + f", tài liệu số {doc_number} | trang {name_equipmentFactory[1][-7:].split(' ')[-1]}"
                        else:
                            COMMENTS[0] = f"Tên công ty sản xuất phù hợp với đơn đăng ký tài liệu số {doc_number} | trang {name_equipmentFactory[1][-7:].split(' ')[-1]}"
                else:
                    name_equipmentFactory = None
                    
            ####
            if not flags[1]:
                if owner_name is not None:
                    name_equipmentOwner = normal_search(vertical_results, horizontal_results, replace_all(owner_name, punc_search), score=60)
                    if name_equipmentOwner[0]:                    
                        comments.append({ 
                                "commentContent" : f"Tên công ty sở hữu phù hợp với đơn đăng ký | trang {name_equipmentOwner[1][-7:].split(' ')[-1]}",
                                "commentStatus" : "OK"
                            })
                        flags[1] = 1

                        if COMMENTS[1]:
                            COMMENTS[1] = COMMENTS[1] + f", tài liệu số {doc_number} | trang {name_equipmentOwner[1][-7:].split(' ')[-1]}"
                        else:
                            COMMENTS[1] = f"Tên công ty chủ sở hữu phù hợp với đơn đăng ký tài liệu số {doc_number} | trang {name_equipmentOwner[1][-7:].split(' ')[-1]}"
                else:
                    name_equipmentOwner = None
                    
            # equipment_flag = 1
            if not flags[2]:
                # equipments = []
                # no_eqt = []
                if eqt_types:
                    for eqt in eqt_types:
                        t_score = normal_search(vertical_results, horizontal_results, replace_all(eqt, punc_search), score=70) if eqt else []
                        # t_score = []
                        # for t_equipment in t_equipments:
                        #     t_score.append(fuzz.token_set_ratio(re.sub(' +', ' ',re.sub(r'[^\w\s]','',t_equipment[:-7].lower())), re.sub(' +', ' ',re.sub(r'[^\w\s]','',eqt.lower()))) if t_equipment else 0)
                        if t_score:
                            if not t_score[0]:
                                no_eqt.append(eqt)
                                no_eqt = set(no_eqt)
                                no_eqt = list(no_eqt)
                            else:
                                try:
                                    no_eqt.remove(eqt)
                                except:
                                    continue
                        
                if not no_eqt and eqt_types:
                    comments.append({
                        "commentContent" : f"Danh sách chủng loại TTBYT đầy đủ",
                        "commentStatus" : "OK"
                    })    
                    flags[2] = 1

                    if COMMENTS[2]:
                        COMMENTS[2] = COMMENTS[2] + f', tài liệu số {doc_number}'
                    else:
                        COMMENTS[2] = f'Danh sách chủng loại TTBYT đầy đủ tài liệu số {doc_number}'
                

            if not flags[3]:
                # no_code = []
                codes = []
                for code in eqt_code:
                    if code:
                        code_score = normal_search(vertical_results, horizontal_results, replace_all(code, punc_search), score=80)
                        
                        # for t_code in t_codes:
                        #     code_score.append(fuzz.token_set_ratio(re.sub(r'[^\w\s]','',t_code[:-7]), re.sub(r'[^\w\s]','',code)))
                        if code_score:
                            if not code_score[0]:                        
                                no_code.append(code)
                                no_code = set(no_code)
                                no_code = list(no_code)
                            else:
                                try:
                                    no_code.remove(code)
                                except:
                                    continue

                if not no_code and eqt_code:
                    comments.append({
                        "commentContent" : f"Danh sách mã TTBYT đầy đủ",
                        "commentStatus" : "OK"
                    })    
                    flags[3] = 1

                    if COMMENTS[3]:
                        COMMENTS[3] = COMMENTS[3] + f', tài liệu số {doc_number}'
                    else:
                        COMMENTS[3] = f'Danh sách mã TTBYT đầy đủ tài liệu số {doc_number}'
                
            if not flags[4]:
                valid_parten = ['expir', 'valid', 'có giá trị đến', 'ngày hết','kết thúc hiệu lực','đến ngày']
                valid_flag = 0
                if check_annot:
                    dates = []
                    punc = {"?":" ", ".":" ", ";":" ", ":":" ", "!":" ", ",":" ","[":" ","]":" ","{":" ","}":" ", "-":"/", "tháng":"/", "năm":"/", "until": "",
                            "juni":"jun", "juli":"jul", "mai":"may", "januar":"jan", "februar":'february', "märz":"mar", "oktober":"oct", "dezember":"dec",
                            "janvier":"jan", "février":"feb", "mars":"mar", "avril":"apr", "juin":"jun", "juillet":"jul", "aout":"aug", "septembre":"sep", "octobre":"oct", "novembre":"nov", "décembre":"dec"}
                    valid_date = []
                    # for result in vertical_results:
                    for dat in vertical_results:
                        # print('dat: ', dat)
                        dat = replace_all(re.sub(' +', ' ',dat.lower()), punc)[:-7]
                        # print('dat: ', dat)
                        try:
                            date = list(datefinder.find_dates(dat.strip(), strict=False))
                            # print('date: ', date)
                            date = datefinder_process(date, dat)
                            # print('date_process: ', date)
                            for patern in valid_parten:
                                if patern in dat:
                                    valid_flag = 1
                            dates.extend(date)
            
                        except:
                            continue
                    if valid_flag:
                        valid_date = dates
                    if valid_date:
                        if max(valid_date) > time_check:
                            comments.append({ 
                                "commentContent": f"Thời gian còn hiệu lực đến ngày {max(valid_date).strftime('%d/%m/%Y')}",
                                "commentStatus" : "OK"
                            })
                            flags[4] = 1
                            COMMENTS[4] = 1
                        else:
                            comments.append({ 
                                "commentContent": f"Thời gian hết hiệu lực đến ngày {max(valid_date).strftime('%d/%m/%Y')}",
                                "commentStatus" : "NOK"
                            })
                            err.append(6)
                    # print('dates: ', dates)
                    if dates and not valid_flag:
                        valid_date = max(dates)
                        begin_date = min(dates)
                        if valid_date > time_check:
                            comments.append({ 
                                "commentContent": f"Thời gian còn hiệu lực đến ngày {valid_date.strftime('%d/%m/%Y')}",
                                "commentStatus" : "OK"
                            })
                            flags[4] = 1
                            COMMENTS[4] = 1

                        if not flags[4]:
                            if begin_date + datetime.timedelta(days=1096) > time_check:
                                comments.append({ 
                                "commentContent": f"Thời gian còn hiệu lực 3 năm từ ngày {begin_date.strftime('%d/%m/%Y')}",
                                "commentStatus" : "OK"
                                })
                                flags[4] = 1
                                COMMENTS[4] = 1
                    if not flags[4] and not valid_flag:
                        comments.append({ 
                                    "commentContent": f"Thời gian hết hiệu lực vào ngày {valid_date.strftime('%d/%m/%Y')}" if valid_date else 'Không tìm thấy thời gian',
                                    "commentStatus" : "NOK"
                                })
                        err.append(6)            

            if not check_annot:
                if not flags[0]:
                    comments.append({ 
                        "commentContent" : "Tên công ty sản xuất không phù hợp với đơn đăng kí",
                        "commentStatus" : "NOK"
                    })
                    err.append(2)
                if not flags[1]:
                    comments.append({
                            "commentContent" : "Tên công ty sở hữu không phù hợp với đơn đăng kí",
                            "commentStatus" : "NOK"
                        })
                    err.append(3)
                if not flags[2]:
                    print('no_eqt: ', no_eqt)
                    comments.append({
                        "commentContent" : f"Danh sách chủng loại TTBYT ko đúng : {', '.join(n for n in no_eqt)}" if no_eqt else f"Danh sách chủng loại TTBYT không được đăng ký",
                        "commentStatus" : "NOK" if no_eqt else "OK"
                    })
                    
                    err.append(4)
                
                if not flags[3]:
                    print(no_code)
                    comments.append({
                        "commentContent" : f"Danh sách mã TTBYT ko đúng : {' ,'.join(n for n in no_code)}" if no_code else f"Danh sách mã TTBYT không được đăng ký",
                        "commentStatus" : "NOK" if no_code else "OK"
                    })
                    err.append(5)

            doc_type = 2

            
        return comments, doc_type, err, flags, COMMENTS, no_eqt, no_code



if __name__ == '__main__':

    cfs = CFS()
    pdf_path = 'test1.pdf'

    es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)

    vertical_results, horizontal_results = cfs.deeplearning_reader(pdf_path)
    print('--'*20)
    print(vertical_results)
    push_result(es, vertical_results)
    query = '760-4528'
    result = search(es,query,1)
    print(result)



    # print(pdf_info)


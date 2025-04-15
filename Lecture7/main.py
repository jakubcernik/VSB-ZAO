import os, cv2, numpy as np, dlib, time
from scipy.spatial import distance
from imutils import face_utils
from concurrent.futures import ProcessPoolExecutor

EAR_THRESHOLD = 0.18
annotation = []
detector = None
predictor = None
leftStart = leftEnd = rightStart = rightEnd = None

def ground_truth(path):
    br = []
    try:
        with open(path,'r') as f:
            for ln in f:
                if ln.strip():
                    s,e = map(int,ln.split())
                    br.append((s,e))
    except: pass
    return br

def ear(eye):
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])
    return (A+B)/(2.0*C)

def init_g():
    global detector,predictor,leftStart,leftEnd,rightStart,rightEnd,annotation
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./landmark_models/dlib_shape_predictor_68_face_landmarks.dat')
    (leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    annotation = ground_truth('./video1/anot1.txt')

def process_frame(args):
    global detector,predictor,leftStart,leftEnd,rightStart,rightEnd,EAR_THRESHOLD,annotation
    imgf, frm, skip = args
    gtc = any(s <= frm <= e for s,e in annotation)
    fr = cv2.imread(os.path.join('video1',imgf))
    sc = 0.75
    fr_s = cv2.resize(fr,(0,0),fx=sc,fy=sc)
    gray = cv2.cvtColor(fr_s,cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    pc = False; ev = 0
    for f in faces:
        sf = dlib.rectangle(int(f.left()/sc),int(f.top()/sc),
                            int(f.right()/sc),int(f.bottom()/sc))
        shp = face_utils.shape_to_np(predictor(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), sf))
        lEye = shp[leftStart:leftEnd]; rEye = shp[rightStart:rightEnd]
        v = (ear(lEye)+ear(rEye))/2.0
        ev = v
        if v<EAR_THRESHOLD: pc=True
        if not skip:
            cv2.drawContours(fr,[cv2.convexHull(lEye)],-1,(0,255,0),1)
            cv2.drawContours(fr,[cv2.convexHull(rEye)],-1,(0,255,0),1)
            if pc: cv2.putText(fr,'MRKNUTI!',(10,30),cv2.FONT_HERSHEY_SIMPLEX,
                               0.7,(0,0,255),2)
            cv2.putText(fr,f'EAR: {v:.2f}',(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,
                        (255,255,255),2)
    if not skip:
        cv2.putText(fr,f'Frame: {frm}',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(fr,f'GT: {"Closed" if gtc else "Open"}',(10,90),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(fr,f'Pred: {"Closed" if pc else "Open"}',(10,120),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0 if pc==gtc else 0),2)
    return {'frm':frm,'gtc':gtc,'pc':pc,'ear':ev,'fr':None if skip else fr}

def main():
    st = time.time()
    global annotation,detector,predictor,leftStart,leftEnd,rightStart,rightEnd
    annotation = ground_truth('./video1/anot1.txt')
    detector = dlib.get_frontal_face_detector()
    lf = './landmark_models/dlib_shape_predictor_68_face_landmarks.dat'

    if not os.path.exists(lf): return
    predictor = dlib.shape_predictor(lf)
    (leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    images = sorted([f for f in os.listdir('./video1') if f.endswith('.jpg')],
                  key=lambda x: int(os.path.splitext(x)[0][3:]))
    process_args = []
    nth_frame = 1
    display_frequency = 10


    for i,imf in enumerate(images[::nth_frame]):
        false_negative = int(os.path.splitext(imf)[0][3:])
        process_args.append((imf,false_negative,(i%display_frequency!=0)))
    rs = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for _ in range(os.cpu_count()):
            ex.submit(init_g)
        for r in ex.map(process_frame, process_args):
            rs.append(r)
            if r['fr'] is not None:
                cv2.imshow('Frame',r['fr'])
                if cv2.waitKey(1)&0xFF==27: break
    cv2.destroyAllWindows()
    smoother_result = []; state='open'; cnt=0; consec_close=2; consec_open=2

    for r in sorted(rs,key=lambda x:x['frm']):
        if state=='open' and r['pc']: cnt+=1;
        elif state=='closed' and not r['pc']: cnt+=1
        else: cnt=0
        if state=='open' and cnt>=consec_close: state='closed'; cnt=0
        elif state=='closed' and cnt>=consec_open: state='open'; cnt=0
        smoother_result .append({'frm':r['frm'],'gc':r['gtc'],'pc':(state=='closed')})
    true_positive=false_positive=false_negative=0

    for r in smoother_result :
        if r['pc'] and r['gc']: true_positive+=1
        elif r['pc'] and not r['gc']: false_positive+=1
        elif not r['pc'] and r['gc']: false_negative+=1

    precision = true_positive/(true_positive+false_positive) if true_positive+false_positive>0 else 0
    recall = true_positive/(true_positive+false_negative) if true_positive+false_negative>0 else 0
    f1_score = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, Time: {time.time()-st:.2f}state')

if __name__=='__main__':
    main()
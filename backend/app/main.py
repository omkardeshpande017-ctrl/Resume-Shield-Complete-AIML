import os,re,uuid,datetime
from pathlib import Path
from typing import Optional
import jwt
from fastapi import FastAPI,UploadFile,File,Form,HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pymongo import MongoClient
from passlib.context import CryptContext
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

BASE=Path(__file__).resolve().parents[1]; UP=BASE/os.getenv('UPLOAD_DIR','uploads'); UP.mkdir(exist_ok=True)
client=MongoClient(os.getenv('MONGO_URL','mongodb://localhost:27017')); db=client[os.getenv('DB_NAME','resume_shield')]
pwd=CryptContext(schemes=['bcrypt'],deprecated='auto'); SECRET=os.getenv('JWT_SECRET','dev-secret-change-me')
app=FastAPI(title='Resume Shield API',version='3.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])

SKILLS={'python','java','javascript','typescript','c','c++','go','golang','sql','mongodb','postgresql','mysql','redis','html','css','react','react.js','node.js','node','express','fastapi','flask','django','next.js','vite','tailwind','docker','kubernetes','aws','azure','gcp','git','github','rest api','graphql','machine learning','deep learning','nlp','tensorflow','pytorch','scikit-learn','pandas','numpy','matplotlib','power bi','tableau','spark','hadoop','figma','linux','firebase','supabase','spring boot','android'}
CATS={'Languages':{'python','java','javascript','typescript','c','c++','go','golang','sql'},'Frontend':{'html','css','react','react.js','typescript','next.js','vite','tailwind'},'Backend':{'node.js','node','express','fastapi','flask','django','spring boot','rest api','graphql'},'Data/ML':{'machine learning','deep learning','nlp','tensorflow','pytorch','scikit-learn','pandas','numpy','matplotlib','spark','hadoop','power bi','tableau'},'Databases':{'mongodb','postgresql','mysql','redis','firebase','supabase'},'DevOps/Cloud':{'docker','kubernetes','aws','azure','gcp','git','github','linux'},'Design':{'figma'}}
HEADINGS=['summary','objective','education','experience','work experience','internship','skills','technical skills','projects','certifications','achievements','languages','publications','contact']
ACTION={'built','developed','designed','implemented','created','led','optimized','automated','deployed','improved','managed','engineered','analyzed','reduced','increased','delivered'}

class AuthIn(BaseModel): email:str; password:str; name:str='Candidate'
class LoginIn(BaseModel): email:str; password:str
class JDIn(BaseModel): resume_id:str; job_description:str
class ImproveIn(BaseModel): resume_id:str; target_role:str='Software Engineer'
class GenerateIn(BaseModel): resume_id:str; title:str='ATS Resume'

def token(uid): return jwt.encode({'sub':uid,'exp':datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=24)},SECRET,algorithm='HS256')
def user_dep(auth:Optional[str]=None):
    if not auth: raise HTTPException(401,'Authentication required')
    try:return db.users.find_one({'_id':auth})
    except: raise HTTPException(401,'Invalid authentication')
from fastapi import Header
def current_user(authorization:str=Header(None)):
    if not authorization or not authorization.startswith('Bearer '): raise HTTPException(401,'Authentication required')
    try:
        p=jwt.decode(authorization[7:],SECRET,algorithms=['HS256']); u=db.users.find_one({'_id':p['sub']})
        if not u: raise Exception()
        return u
    except: raise HTTPException(401,'Invalid or expired token')

def extract(path):
    if path.suffix.lower()=='.pdf': return '\n'.join((p.extract_text() or '') for p in PdfReader(str(path)).pages)
    d=Document(str(path)); return '\n'.join(x.text for x in d.paragraphs)+'\n'+'\n'.join(' | '.join(c.text for c in row.cells) for t in d.tables for row in t.rows)
def clean(t): return re.sub(r'\s+',' ',t).strip()
def skills(t):
    low=t.lower(); return sorted([s for s in SKILLS if re.search(r'(?<![\w+#])'+re.escape(s)+r'(?![\w+#])',low)])
def sections(t):
    out={}; lines=t.splitlines(); cur='general'
    for line in lines:
        s=line.strip(); k=re.sub(r'[^a-z ]','',s.lower())
        if k in HEADINGS or any(k==h for h in HEADINGS): cur=k; out.setdefault(cur,[]); continue
        if s: out.setdefault(cur,[]).append(s)
    return {k:' '.join(v) for k,v in out.items()}
def ats(t,ss,sk):
    low=t.lower(); words=re.findall(r'\b\w+\b',t); n=len(words); bullets=len(re.findall(r'(^|\n)\s*[•●▪*-]\s+',t)); nums=len(re.findall(r'\b\d+(?:\.\d+)?%?\b',t)); verbs=sum(1 for x in ACTION if re.search(r'\b'+x+r'\b',low));
    checks={'contact_info':bool(re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+',t)),'headings':sum(1 for h in HEADINGS if h in ss)>=4,'skills_section':('skills' in ss or 'technical skills' in ss),'projects':('projects' in ss),'education':('education' in ss),'experience':('experience' in ss or 'work experience' in ss),'quantified_achievements':nums>=3,'action_verbs':verbs>=3,'reasonable_length':250<=n<=1200,'no_graphics_only':True,'readable_structure':len(ss)>=4}
    score=round(sum(checks.values())/len(checks)*100)
    return score,checks,{'word_count':n,'bullet_count':bullets,'quantified_count':nums,'action_verb_count':verbs}
def extract_identity(t):
    email=re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+',t); phone=re.search(r'(?:\+?\d[\d ()-]{8,}\d)',t); urls=re.findall(r'https?://[^\s)]+',t)
    first=next((x.strip() for x in t.splitlines() if x.strip()),'Candidate')
    return {'name':first[:80],'email':email.group(0) if email else '', 'phone':phone.group(0) if phone else '', 'links':urls[:10]}
def analyze(t):
    ss=sections(t); sk=skills(t); a,checks,metrics=ats(t,ss,sk); words=len(re.findall(r'\b\w+\b',t));
    edu=20 if 'education' in ss else 0; exp=20 if ('experience' in ss or 'work experience' in ss) else 0; proj=20 if 'projects' in ss else 0; skillscore=min(100,len(sk)*7); content=min(100,40+(metrics['quantified_count']*8)+(metrics['action_verb_count']*5)); complete=round(sum(bool(x) for x in [extract_identity(t)['email'],extract_identity(t)['phone'],sk,edu,exp,proj])/6*100)
    return {'ats':a,'skills':skillscore,'experience':exp*5,'education':edu*5,'projects':proj*5,'content_quality':content,'completeness':complete,'formatting':round(a*.9),'metrics':metrics,'checks':checks,'skills_list':sk,'skill_groups':{k:sorted(set(sk)&v) for k,v in CATS.items()},'sections':list(ss.keys())}
def overall(x,match=0,integrity=100): return round(.20*x['ats']+.25*match+.20*x['content_quality']+.15*x['skills']+.10*x['completeness']+.10*integrity) if match else round(.35*x['ats']+.25*x['content_quality']+.20*x['skills']+.20*x['completeness'])

def match(res,jd):
    a=skills(res); b=skills(jd); vec=TfidfVectorizer(stop_words='english').fit_transform([res,jd]); sem=round(float(cosine_similarity(vec[0:1],vec[1:2])[0][0])*100); cov=round(len(set(a)&set(b))/max(1,len(set(b)))*100); common=sorted(set(a)&set(b)); missing=sorted(set(b)-set(a)); score=round(.55*sem+.30*cov+.15*min(100, len(set(re.findall(r'\b[a-zA-Z]{5,}\b',res.lower()))) / max(1,len(set(re.findall(r'\b[a-zA-Z]{5,}\b',jd.lower()))))*100))
    return {'match_percent':score,'semantic_similarity':sem,'skill_coverage':cov,'matching_skills':common,'missing_skills':missing,'recommended_skills':missing[:10]}

def doc(uid,rid,title):
    r=db.resumes.find_one({'_id':rid,'user_id':uid}); path=UP/r['filename']; text=extract(path); lines=text.splitlines(); out=UP/f'{uuid.uuid4()}.pdf'; styles=getSampleStyleSheet(); story=[Paragraph(title,styles['Title']),Spacer(1,10)]
    for line in lines:
        if line.strip(): story += [Paragraph(line.replace('&','&amp;'),styles['BodyText']),Spacer(1,5)]
    SimpleDocTemplate(str(out),pagesize=A4).build(story); return out

@app.get('/api/health')
def health():
    try: db.command('ping'); mongo=True
    except: mongo=False
    return {'status':'ok','database':mongo,'ml':'TF-IDF + rule-based NLP'}
@app.post('/api/auth/register')
def register(x:AuthIn):
    if db.users.find_one({'email':x.email.lower()}): raise HTTPException(409,'Email already registered')
    uid=str(uuid.uuid4()); db.users.insert_one({'_id':uid,'name':x.name,'email':x.email.lower(),'password':pwd.hash(x.password),'created_at':datetime.datetime.utcnow()}); return {'token':token(uid),'user':{'id':uid,'name':x.name,'email':x.email.lower()}}
@app.post('/api/auth/login')
def login(x:LoginIn):
    u=db.users.find_one({'email':x.email.lower()})
    if not u or not pwd.verify(x.password,u['password']): raise HTTPException(401,'Invalid email or password')
    return {'token':token(u['_id']),'user':{'id':u['_id'],'name':u['name'],'email':u['email']}}
@app.get('/api/dashboard')
def dashboard(u=Depends(current_user)):
    rs=list(db.resumes.find({'user_id':u['_id']},{'text':0})); analyses=list(db.analyses.find({'user_id':u['_id']}).sort('created_at',-1).limit(20));
    return {'user':{'name':u['name'],'email':u['email']},'resume_count':len(rs),'analysis_count':len(analyses),'history':[{'id':a['_id'],'resume_id':a['resume_id'],'overall':a.get('overall',0),'created_at':a['created_at']} for a in analyses]}
@app.post('/api/resume/upload')
def upload(file:UploadFile=File(...),u=Depends(current_user)):
    ext=Path(file.filename).suffix.lower()
    if ext not in ['.pdf','.docx']: raise HTTPException(400,'Only PDF and DOCX are supported')
    if file.size and file.size>10*1024*1024: raise HTTPException(400,'Maximum file size is 10MB')
    rid=str(uuid.uuid4()); name=rid+ext; data=file.file.read(); (UP/name).write_bytes(data); text=extract(UP/name); parsed=analyze(text); identity=extract_identity(text); doc={'_id':rid,'user_id':u['_id'],'filename':name,'original_name':file.filename,'text':text,'identity':identity,'analysis':parsed,'created_at':datetime.datetime.utcnow()}; db.resumes.insert_one(doc); aid=str(uuid.uuid4()); db.analyses.insert_one({'_id':aid,'user_id':u['_id'],'resume_id':rid,'overall':overall(parsed),'analysis':parsed,'created_at':datetime.datetime.utcnow()}); return {'resume_id':rid,'filename':file.filename,'identity':identity,'analysis':parsed,'overall_score':overall(parsed)}
@app.get('/api/resume/{rid}')
def get_resume(rid:str,u=Depends(current_user)):
    r=db.resumes.find_one({'_id':rid,'user_id':u['_id']},{'text':0});
    if not r: raise HTTPException(404,'Resume not found')
    return r
@app.post('/api/resume/{rid}/analyze')
def reanalyze(rid:str,u=Depends(current_user)):
    r=db.resumes.find_one({'_id':rid,'user_id':u['_id']});
    if not r: raise HTTPException(404,'Resume not found')
    x=analyze(r['text']); db.resumes.update_one({'_id':rid},{'$set':{'analysis':x}}); return {'analysis':x,'overall_score':overall(x)}
@app.post('/api/resume/{rid}/ats')
def ats_route(rid:str,u=Depends(current_user)):
    r=db.resumes.find_one({'_id':rid,'user_id':u['_id']});
    if not r: raise HTTPException(404,'Resume not found')
    x=analyze(r['text']); return {'score':x['ats'],'checks':x['checks'],'metrics':x['metrics']}
@app.post('/api/resume/{rid}/match')
def match_route(rid:str,x:JDIn,u=Depends(current_user)):
    r=db.resumes.find_one({'_id':rid,'user_id':u['_id']});
    if not r: raise HTTPException(404,'Resume not found')
    m=match(r['text'],x.job_description); a=r['analysis']; ov=overall(a,m['match_percent']); db.analyses.insert_one({'_id':str(uuid.uuid4()),'user_id':u['_id'],'resume_id':rid,'overall':ov,'match':m,'created_at':datetime.datetime.utcnow()}); return {'overall_score':ov,'match':m}
@app.post('/api/integrity/check')
def integrity(x:ImproveIn,u=Depends(current_user)):
    r=db.resumes.find_one({'_id':x.resume_id,'user_id':u['_id']});
    if not r: raise HTTPException(404,'Resume not found')
    t=r['text']; links=extract_identity(t)['links']; issues=[]
    for l in links:
        if any(s in l.lower() for s in ['linkedin.com','github.com','gitlab.com','portfolio']): continue
        issues.append({'type':'external_link','severity':'review','message':f'Unrecognized external link: {l}'})
    certs=re.findall(r'(?im)(?:certificate|certification)[:\- ]+([^\n]{3,80})',t); projects=re.findall(r'(?im)(?:project)[:\- ]+([^\n]{3,100})',t)
    return {'integrity_score':max(0,100-len(issues)*10),'signals':{'links_found':len(links),'certification_mentions':len(certs),'project_mentions':len(projects)},'issues':issues,'note':'Integrity checks are verification signals, not proof of fraud.'}
@app.post('/api/scam/check')
def scam(x:JDIn,u=Depends(current_user)):
    t=x.job_description.lower(); red=[]
    patterns=[('payment_request',r'pay\s+(?:a|the)?\s*(?:fee|money|deposit|registration)'),('guaranteed_job',r'guaranteed\s+(?:job|income|placement)'),('bank_details',r'bank\s+(?:account|details)|credit card'),('crypto',r'crypto|bitcoin|usdt'),('urgent',r'urgent|immediately|act now')]
    for n,p in patterns:
        if re.search(p,t): red.append({'signal':n,'severity':'high' if n in ['payment_request','bank_details'] else 'medium'})
    score=max(0,100-len(red)*18); return {'safety_score':score,'risk_level':'high' if score<50 else 'medium' if score<80 else 'low','signals':red,'disclaimer':'This is a screening aid, not a definitive determination of a job scam.'}
@app.post('/api/resume/{rid}/improve')
def improve(rid:str,x:ImproveIn,u=Depends(current_user)):
    r=db.resumes.find_one({'_id':rid,'user_id':u['_id']});
    if not r: raise HTTPException(404,'Resume not found')
    a=r['analysis']; suggestions=[]
    if a['metrics']['quantified_count']<3: suggestions.append({'area':'Impact','before':'Worked on a project.','after':'Developed and delivered a project, improving measurable performance by 20%.','reason':'Add a metric to demonstrate impact.'})
    if a['metrics']['action_verb_count']<3: suggestions.append({'area':'Action verbs','before':'Responsible for tasks.','after':'Implemented and optimized core tasks.','reason':'Start bullets with strong action verbs.'})
    if not a['checks']['contact_info']: suggestions.append({'area':'Contact','before':'Missing contact details','after':'Add a professional email and phone number.','reason':'Recruiters need a clear contact path.'})
    if not a['checks']['headings']: suggestions.append({'area':'Structure','before':'Inconsistent sections','after':'Use clear headings: Summary, Skills, Experience, Projects, Education.','reason':'Improves ATS parsing.'})
    if not suggestions: suggestions.append({'area':'Polish','before':'Current bullets','after':'Keep bullets concise and quantified; tailor keywords to the target role.','reason':'Maintain ATS readability and relevance.'})
    return {'target_role':x.target_role,'suggestions':suggestions,'ai_engine':'Local explainable NLP/ML'}
@app.post('/api/resume/generate')
def generate(x:GenerateIn,u=Depends(current_user)):
    try: p=doc(u['_id'],x.resume_id,x.title); return {'download_url':'/api/generated/'+p.name}
    except Exception as e: raise HTTPException(500,str(e))
@app.get('/api/generated/{name}')
def generated(name:str):
    p=UP/name
    if not p.exists(): raise HTTPException(404,'File not found')
    return FileResponse(str(p),media_type='application/pdf',filename='resume-shield-generated.pdf')

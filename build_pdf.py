import re, os, hashlib, matplotlib
matplotlib.rcParams["mathtext.fontset"]="cm"; matplotlib.rcParams["font.family"]="serif"
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import pypdfium2 as pdfium
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Image, Table, TableStyle, KeepTogether, FrameBreak, NextPageTemplate, HRFlowable)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

tex=open("rcp_paper.tex").read()

# ---------- 0. classical scientific-plate palette ----------
# minimalist, old-school, serif, Newtonian. Restrained contrast, semantic accents only.
PARCH   = "#FFFFFF"   # white ground (monochrome)
INK     = "#000000"   # black ink (body)
CHARCOAL= "#000000"   # section headings (black)
WARMGRAY= "#666666"   # footer / secondary (neutral gray)
PALEGRAY= "#BEBEBE"   # hairlines, figure keylines (neutral gray)
SEPIA   = "#000000"   # accent -> black (structural rules / labels)
GREENGRAY="#000000"   # near -> black
REDBROWN= "#666666"   # far -> mid gray
THMTINT = "#F0F0F0"   # theorem block ground (very light neutral)
PROOFINK= "#333333"   # proof body (dark gray)

# ---------- 1. render equations ----------
eqs=re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", tex, re.S)
def prep(s):
    s=re.sub(r"\\label\{[^}]*\}","",s)
    s=s.replace("\\Ksig","K_\\sigma").replace("\\smax","\\sigma_{\\mathrm{max}}")
    s=s.replace("\\sstar","\\sigma^{*}").replace("\\Neff","N_{\\mathrm{eff}}").replace("\\Rbb","\\mathbb{R}")
    s=s.replace("\\lVert","\\|").replace("\\rVert","\\|")
    for m in ["\\!","\\,","\\;","\\quad","\\qquad"]: s=s.replace(m," ")
    s=s.replace("\\boxed{","{"); s=s.replace("\\left","").replace("\\right","")
    for b in ["\\Big(","\\big("]: s=s.replace(b,"(")
    for b in ["\\Big)","\\big)"]: s=s.replace(b,")")
    s=s.replace("\\tfrac","\\frac").replace("\\dfrac","\\frac").replace("\\ge","\\geq").replace("\\le","\\leq")
    s=s.replace("\\geqq","\\geq").replace("\\leqq","\\leq")
    s=s.replace("\\sigma^{*}^2","\\sigma^{*2}").replace("^{*}^2","^{*2}")
    # strip \underbrace{X}_{...} keeping X, handling nested braces in the subscript
    while "\\underbrace" in s:
        k=s.find("\\underbrace"); j=k+len("\\underbrace")
        if j>=len(s) or s[j]!="{": break
        depth=1; p=j+1
        while p<len(s) and depth:
            if s[p]=="{":depth+=1
            elif s[p]=="}":depth-=1
            p+=1
        inner=s[j+1:p-1]  # X
        # now expect _{...}
        rest=s[p:]
        if rest.startswith("_{"):
            depth=1; q=p+2
            while q<len(s) and depth:
                if s[q]=="{":depth+=1
                elif s[q]=="}":depth-=1
                q+=1
            s=s[:k]+inner+s[q:]
        else:
            s=s[:k]+inner+s[p:]
    s=re.sub(r"\s+"," ",s).strip().rstrip(",.")
    return s
os.makedirs("eqimg",exist_ok=True)
EQ_DPI=300
_EQC=(0.10,0.10,0.10)  # ink for equation glyphs
for i,e in enumerate(eqs):
    fig=plt.figure(figsize=(0.01,0.01)); fig.text(0,0,f"${prep(e)}$",fontsize=16,color=_EQC)
    fig.savefig(f"eqimg/eq{i}.png",dpi=EQ_DPI,bbox_inches="tight",pad_inches=0.05,facecolor=PARCH); plt.close(fig)
print("equations rendered:",len(eqs))
# meta-title glyph: big sigma* set in the same math engine as the body equations
_mf=plt.figure(figsize=(0.01,0.01)); _mf.text(0,0,r"$\sigma^{*}$",fontsize=54,color=_EQC)
_mf.savefig("meta_sigma.png",dpi=EQ_DPI,bbox_inches="tight",pad_inches=0.02,facecolor=PARCH); plt.close(_mf)
# uniform on-page scale: every displayed equation renders at the SAME font size.
# pick the largest scale at which the widest equation still fits the column.
from reportlab.lib.pagesizes import A4 as _A4
from reportlab.lib.units import mm as _mm
_PW,_PH=_A4; _M=16*_mm; _CG=7*_mm; _COLW=(_PW-2*_M-_CG)/2
_base=72.0/EQ_DPI
_maxw=max(PILImage.open(f"eqimg/eq{i}.png").size[0] for i in range(len(eqs))) if eqs else 1
EQ_SCALE=min(1.0, ((_COLW-8)*0.96)/(_maxw*_base))
print("uniform equation scale:",round(EQ_SCALE,3))

# ---------- 2. label map + inline converter ----------
body=tex.split("\\end{@twocolumnfalse}\n]",1)[1].split("\\end{document}")[0]
abstract=re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
_USUP={"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","+":"⁺","-":"⁻","*":"*","n":"ⁿ","i":"ⁱ"}
_USUB={"0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉","+":"₊","-":"₋","i":"ᵢ","j":"ⱼ","r":"ᵣ","e":"ₑ","f":"f"}
def _u(s,sup):
    d=_USUP if sup else _USUB; return "".join(d.get(c,c) for c in s)
def _flatten(inner):
    inner=re.sub(r"\^\{([^{}]*)\}", lambda m:_u(m.group(1),True), inner)
    inner=re.sub(r"_\{([^{}]*)\}", lambda m:_u(m.group(1),False), inner)
    inner=re.sub(r"\^(\w|\*)", lambda m:_u(m.group(1),True), inner)
    inner=re.sub(r"_(\w)", lambda m:_u(m.group(1),False), inner); return inner
def render_scripts(s):
    out=[]; i=0; L=len(s)
    while i<L:
        c=s[i]
        if c in "^_":
            tag="super" if c=="^" else "sub"; i+=1
            if i<L and s[i]=="{":
                depth=1; j=i+1
                while j<L and depth:
                    if s[j]=="{":depth+=1
                    elif s[j]=="}":depth-=1
                    j+=1
                inner=s[i+1:j-1]; i=j
            elif i<L: inner=s[i]; i+=1
            else: inner=""
            out.append(f"<{tag}>{_flatten(inner)}</{tag}>")
        else: out.append(c); i+=1
    return "".join(out)
def protect_tags(s): return re.sub(r"<(/?(?:b|i|sub|super)|font[^>]*|/font)>", lambda m:"\x00%s\x01"%m.group(1), s)
def restore_tags(s):
    s=s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"); return s.replace("\x00","<").replace("\x01",">")
def units(m):
    val=m.group(1); u=m.group(2).replace("\\percent","%").replace("\\times","×"); return val+re.sub(r"\\[a-zA-Z]+","",u)
def inline_math(m):
    s=m.group(1)
    s=re.sub(r"\\leq?(?![a-zA-Z])","≤",s); s=re.sub(r"\\geq?(?![a-zA-Z])","≥",s)
    s=s.replace("\\smax","σ_{max}").replace("\\sstar","σ^{*}").replace("\\Ksig","K_{σ}")
    s=s.replace("\\Neff","N_{eff}").replace("\\varepsilon","ε").replace("\\epsilon","ε")
    for k,v in {r"\\sigma":"σ",r"\\alpha":"α",r"\\rho":"ρ",r"\\lambda":"λ",r"\\kappa":"κ",r"\\mu":"μ",r"\\omega":"ω",r"\\gamma":"γ",r"\\beta":"β",r"\\delta":"δ",r"\\times":"×",r"\\to":"→",r"\\infty":"∞",r"\\in\b":"∈",r"\\notin":"∉",r"\\approx":"≈",r"\\cdot":"·",r"\\pm":"±",r"\\Rbb":"R",r"\\mathbb\{R\}":"R",r"\\sum":"Σ",r"\\ldots":"…",r"\\star":"*",r"\\partial":"∂",r"\\propto":"∝",r"\\neq":"≠",r"\\forall":"∀",r"\\exists":"∃",r"\\subset":"⊂",r"\\equiv":"≡",r"\\lVert":"||",r"\\rVert":"||"}.items():
        s=re.sub(k,v,s)
    s=re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    s=re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s=s.replace("\\max","max").replace("\\min","min").replace("\\ln","ln").replace("\\log","log")
    s=re.sub(r"\\(mathrm|text|mathcal|mathbf|operatorname)\{([^{}]*)\}", r"\2", s)
    s=re.sub(r"\\[a-zA-Z]+","",s); s=render_scripts(s)
    s=s.replace("{","").replace("}","").replace("\\",""); return "<i>"+s+"</i>"
labelmap={}; sec_counter=0; subsec=0
envc={"theorem":0,"proposition":0,"corollary":0,"definition":0,"remark":0}
envname={"theorem":"Theorem","proposition":"Proposition","corollary":"Corollary","definition":"Definition","remark":"Remark"}
fig_counter=0; eq_counter=0; in_appendix=False; app_sec=0; pending=None
scan=tex.split("\\maketitle",1)[-1]
pat=re.compile(r"\\(section|subsection)\*?\{|\\begin\{(theorem|proposition|corollary|definition|remark|figure\*?|equation|table\*?)\}|\\label\{([^}]+)\}|\\appendix")
tab_counter=0
for m in pat.finditer(scan):
    if m.group(0)=="\\appendix": in_appendix=True; continue
    if m.group(1) in ("section","subsection"):
        if m.group(1)=="section":
            if in_appendix: app_sec+=1; pending=("sec",chr(ord('A')+app_sec-1))
            else: sec_counter+=1; subsec=0; pending=("sec",str(sec_counter))
        else: subsec+=1; pending=("sec",f"{sec_counter}.{subsec}")
    elif m.group(2):
        env=m.group(2)
        if env.startswith("figure"): fig_counter+=1; pending=("fig",str(fig_counter))
        elif env.startswith("table"): tab_counter+=1; pending=("tab",str(tab_counter))
        elif env=="equation": eq_counter+=1; pending=("eq",str(eq_counter))
        else: envc[env]+=1; pending=(env,f"{envname[env]} {envc[env]}")
    elif m.group(3) and pending: labelmap[m.group(3)]=pending[1]
numbermap={}
for k,v in labelmap.items():
    _mm=re.match(r"(Theorem|Proposition|Corollary|Definition|Remark)\s+(.+)",v); numbermap[k]=_mm.group(2) if _mm else v
def _inline_core(t):
    # house style: em dash renders as literal "--" (protect from the en-dash pass via sentinel)
    t=t.replace("\\noindent","").replace("\\%","%").replace("~"," ").replace("---","\x02").replace("--","–").replace("\x02","--")
    t=re.sub(r"\\SI\{([^}]*)\}\{([^}]*)\}", units, t)
    t=re.sub(r"\\numrange\{([^}]*)\}\{([^}]*)\}", r"\1–\2", t); t=re.sub(r"\\num\{([^}]*)\}", r"\1", t)
    for _a,_b in [("\\leq","≤"),("\\le","≤"),("\\geq","≥"),("\\ge","≥"),("\\varepsilon","ε"),("\\epsilon","ε"),
                  ("\\sigma","σ"),("\\times","×"),("\\approx","≈"),("\\,"," "),("\\;"," "),("\\!","")]:
        t=t.replace(_a,_b)
    t=re.sub(r"\\eqref\{([^}]+)\}", lambda m:"Eq. ("+numbermap.get(m.group(1),"?")+")", t)
    t=re.sub(r"\\ref\{([^}]+)\}", lambda m:numbermap.get(m.group(1),"?"), t)
    t=re.sub(r"\\cite\{[^}]*\}", "", t); t=re.sub(r"\\label\{[^}]*\}", "", t)
    t=re.sub(r"\\textbf\{([^}]*)\}", r"<b>\1</b>", t); t=re.sub(r"\\emph\{([^}]*)\}", r"<i>\1</i>", t)
    t=re.sub(r"\\texttt\{([^}]*)\}", r"<font face='Courier'>\1</font>", t); t=re.sub(r"\\textsc\{([^}]*)\}", r"\1", t)
    t=re.sub(r"\$([^$]*)\$", inline_math, t)
    t=t.replace("\\&","&").replace("\\_","_").replace("\\#","#").replace("\\ "," ")
    t=re.sub(r"\s+"," ",t).strip(); return restore_tags(protect_tags(t))
_units={"percent":"%","times":"×","":""}
def inline(s):
    def _sirange(m):
        a,b,u=m.group(1),m.group(2),m.group(3)
        u=u.replace("\\percent","%").replace("\\times","×").strip()
        return f"{a}–{b}{u}" if u in ("%","×") else (f"{a}–{b} {u}" if u else f"{a}–{b}")
    s=re.sub(r"\\SIrange\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", _sirange, s)
    return _inline_core(s)
print("labels:",len(labelmap),"| prop:sigmastar ->",numbermap.get("prop:sigmastar"),"| fig:e0 ->",numbermap.get("fig:e0"))

# ---------- 3. styles ----------
PW,PH=A4; MARGIN=16*mm; COLGAP=7*mm; COLW=(PW-2*MARGIN-COLGAP)/2
S=getSampleStyleSheet()
_INK=colors.HexColor(INK); _CHAR=colors.HexColor(CHARCOAL); _SEP=colors.HexColor(SEPIA)
_GG=colors.HexColor(GREENGRAY); _WG=colors.HexColor(WARMGRAY)
body_st=ParagraphStyle("body",parent=S["Normal"],fontName="Times-Roman",fontSize=9.2,leading=11.2,alignment=TA_JUSTIFY,spaceAfter=2.4,textColor=_INK)
# section headings: small-caps feel via tracked bold charcoal + a thin sepia rule above (drawn in flowable)
h1_st=ParagraphStyle("h1",parent=body_st,fontName="Times-Bold",fontSize=10.8,leading=13,spaceBefore=8,spaceAfter=3,textColor=_CHAR,tracking=1.2)
h2_st=ParagraphStyle("h2",parent=body_st,fontName="Times-Bold",fontSize=9.9,spaceBefore=5,spaceAfter=2,textColor=_CHAR,tracking=0.8)
h3_st=ParagraphStyle("h3",parent=body_st,fontName="Times-BoldItalic",fontSize=9.4,spaceBefore=4,spaceAfter=1.5,textColor=colors.HexColor("#000000"))
thm_st=ParagraphStyle("thm",parent=body_st,fontName="Times-Italic",fontSize=9.2,leading=11.2,leftIndent=7,rightIndent=3,spaceBefore=2.5,spaceAfter=2.5,backColor=colors.HexColor(THMTINT),borderColor=_SEP,borderWidth=0,borderPadding=(4,4,4,7))
proof_st=ParagraphStyle("proof",parent=body_st,fontSize=8.9,leading=10.7,textColor=colors.HexColor(PROOFINK),spaceAfter=2.2)
cap_st=ParagraphStyle("cap",parent=body_st,fontSize=8.1,leading=9.8,textColor=colors.HexColor("#1A1A1A"),spaceBefore=2.5,spaceAfter=4,alignment=TA_LEFT)
bib_st=ParagraphStyle("bib",parent=body_st,fontSize=7.5,leading=8.2,spaceAfter=0.6,leftIndent=6,firstLineIndent=-6,textColor=colors.HexColor("#1A1A1A"))
list_st=ParagraphStyle("list",parent=body_st,leftIndent=10,firstLineIndent=-6,spaceAfter=2)
# meta-title glyph sits above; this is now the SUBTITLE line
title_st=ParagraphStyle("title",parent=body_st,fontName="Times-Roman",fontSize=12.5,leading=15.5,alignment=TA_CENTER,spaceBefore=2,spaceAfter=7,textColor=_CHAR,tracking=0.4)
auth_st=ParagraphStyle("auth",parent=body_st,fontSize=9.5,alignment=TA_CENTER,spaceAfter=2,textColor=_INK)
aff_st=ParagraphStyle("aff",parent=auth_st,fontSize=8,fontName="Times-Italic",spaceAfter=6,textColor=_WG)
abs_head=ParagraphStyle("abh",parent=auth_st,fontName="Times-Bold",fontSize=8.6,spaceBefore=4,spaceAfter=2,textColor=_SEP,tracking=1.5)
abs_st=ParagraphStyle("abs",parent=body_st,fontSize=9.0,leading=11.0,alignment=TA_JUSTIFY,leftIndent=12,rightIndent=12,spaceAfter=4)
THMNAME=envname
def eq_flowable(idx,maxw):
    fn=f"eqimg/eq{idx}.png"; iw,ih=PILImage.open(fn).size
    # uniform scale so all displayed equations share one font size; clamp only if a
    # single equation is still too wide for its column at that scale.
    scale=EQ_SCALE*72.0/EQ_DPI; w,h=iw*scale,ih*scale
    if w>maxw*0.97: s2=(maxw*0.97)/w; w*=s2; h*=s2
    img=Image(fn,width=w,height=h); img.hAlign="CENTER"; return [Spacer(1,2),img,Spacer(1,2.5)]
def fig_flowable(fn,cap,maxw,figno):
    iw,ih=PILImage.open(fn).size; w=(maxw-6)*0.90; h=ih*(w/iw)
    img=Image(fn,width=w,height=h); img.hAlign="CENTER"
    plate=Table([[img]],colWidths=[maxw])
    plate.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(PARCH)),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3)]))
    capp=Paragraph(f"<b><font color='{SEPIA}'>Figure {figno}.</font></b> {inline(cap)}",cap_st)
    return [Spacer(1,2.5),plate,capp]
_ctr={k:0 for k in THMNAME}
def thm_flowable(env,title,content):
    _ctr[env]+=1; label=f"{THMNAME[env]} {_ctr[env]}"
    if title: label+=f" ({inline(title)})"
    flows=[]; head=f"<b>{label}.</b> "
    for p in re.split(r"(@@EQ\d+@@)",content):
        me=re.match(r"@@EQ(\d+)@@",p)
        if me: flows+=eq_flowable(int(me.group(1)),COLW-8)
        elif p.strip(): flows.append(Paragraph(head+inline(p.strip()),thm_st)); head=""
    return flows
def proof_flowable(content):
    flows=[]; head="<i>Proof.</i> "; last_txt=None
    for p in re.split(r"(@@EQ\d+@@)",content):
        me=re.match(r"@@EQ(\d+)@@",p)
        if me: flows+=eq_flowable(int(me.group(1)),COLW-8)
        elif p.strip():
            para=Paragraph(head+inline(p.strip()),proof_st); head=""; flows.append(para); last_txt=para
    if last_txt is not None:
        idx=flows.index(last_txt); flows[idx]=Paragraph(last_txt.text+" ∎",proof_st)
    return flows

# ---------- 4. parse body into blocks ----------
gi=[0]
body_tok=re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", lambda m:f"@@EQ{gi.__setitem__(0,gi[0]+1) or (gi[0]-1)}@@", body, flags=re.S)
blocks=[]; buf=[]
def flush():
    global buf
    txt=" ".join(buf).strip(); buf=[]
    if txt: blocks.append(("para",txt))
lines=body_tok.split("\n"); i=0; n=len(lines)
while i<n:
    s=lines[i].strip()
    if s.startswith("%") or s=="":
        if s=="": flush()
        i+=1; continue
    mo=re.match(r"\\section\*?\{(.+?)\}", s)
    if mo: flush(); blocks.append(("h1",mo.group(1))); i+=1; continue
    mo=re.match(r"\\subsection\*?\{(.+?)\}", s)
    if mo: flush(); blocks.append(("h2",mo.group(1))); i+=1; continue
    mo=re.match(r"\\paragraph\{(.+?)\}(.*)", s)
    if mo:
        flush(); blocks.append(("h3",mo.group(1)))
        if mo.group(2).strip(): buf.append(mo.group(2).strip())
        i+=1; continue
    if s.startswith("\\begin{figure"):
        flush(); j=i; cur=None; cap=""
        while j<n and "\\end{figure" not in lines[j]:
            gm=re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", lines[j]); cm=re.search(r"\\caption\{(.+)", lines[j])
            if gm: cur=gm.group(1)
            if cm:
                cap=cm.group(1); k=j
                while cap.count("{")+1>cap.count("}") and k+1<n: k+=1; cap+=" "+lines[k].strip()
                cap=cap.rsplit("}",1)[0]
            j+=1
        blocks.append(("fig",cur,cap)); i=j+1; continue
    if s.startswith("\\begin{table"):
        flush(); tbl_lines=[s]; i+=1
        while i<n and "\\end{table" not in lines[i]:
            tbl_lines.append(lines[i]); i+=1
        tbl_lines.append(lines[i] if i<n else "")
        blocks.append(("table_alpha","\n".join(tbl_lines)))
        i+=1; continue
    if re.match(r"\\begin\{(theorem|proposition|corollary|definition)\}", s):
        flush(); kind=re.match(r"\\begin\{(\w+)\}(?:\[(.+?)\])?", s); env=kind.group(1); ttl=kind.group(2) or ""
        content=[re.sub(r"\\begin\{\w+\}(\[.+?\])?","",s)]; i+=1
        while i<n and f"\\end{{{env}}}" not in lines[i]:
            if not lines[i].strip().startswith("%"): content.append(lines[i])
            i+=1
        i+=1; blocks.append(("thm",env,ttl," ".join(content).strip())); continue
    if s.startswith("\\begin{proof}"):
        flush(); content=[re.sub(r"\\begin\{proof\}(\[.+?\])?","",s)]; i+=1
        while i<n and "\\end{proof}" not in lines[i]:
            if not lines[i].strip().startswith("%"): content.append(lines[i])
            i+=1
        i+=1; blocks.append(("proof"," ".join(content).strip())); continue
    if re.match(r"\\begin\{(enumerate|itemize)\}", s):
        flush(); env="enumerate" if "enumerate" in s else "itemize"; items=[]; cur=""; i+=1
        while i<n and f"\\end{{{env}}}" not in lines[i]:
            ls=lines[i].strip()
            if ls.startswith("\\item"):
                if cur.strip(): items.append(cur.strip())
                cur=ls.replace("\\item","",1)
            elif not ls.startswith("%"): cur+=" "+ls
            i+=1
        if cur.strip(): items.append(cur.strip())
        i+=1; blocks.append(("list",env,items)); continue
    if s.startswith("\\begin{thebibliography}"):
        flush(); refs=[]; cur=""; i+=1
        while i<n and "\\end{thebibliography}" not in lines[i]:
            ls=lines[i].strip()
            if ls.startswith("\\bibitem"):
                if cur.strip(): refs.append(cur.strip())
                cur=re.sub(r"\\bibitem\{[^}]*\}","",ls)
            elif ls and not ls.startswith("%") and not ls.startswith("\\small"): cur+=" "+ls
            i+=1
        if cur.strip(): refs.append(cur.strip())
        i+=1; blocks.append(("bib",refs)); continue
    if s.startswith("\\appendix"): i+=1; continue
    buf.append(s); i+=1
flush()
for k,b in enumerate(blocks):
    if b[0]=="h1" and "\\ref" in b[1]: blocks[k]=("h1","Proof of Theorem "+numbermap.get("thm:rule","2"))

# ---------- 5. table (per-block: each \begin{table} parses its OWN tabular + caption) ----------
def clean(c):
    c=re.sub(r"\\textbf\{([^}]*)\}",r"\1",c).replace("\\%","%")
    # map known math tokens to glyphs BEFORE stripping residual commands,
    # else e.g. $\sstar$ -> blank and tail/$\varepsilon$ -> "tail/"
    for _a,_b in [("\\sstar","σ*"),("\\smax","σmax"),("\\varepsilon","ε"),("\\epsilon","ε"),
                  ("\\times","×"),("\\leq","≤"),("\\le","≤"),("\\geq","≥"),("\\ge","≥"),
                  ("\\Neff","Neff"),("\\pm","±")]:
        c=c.replace(_a,_b)
    c=c.replace("$","")
    c=re.sub(r"\\[a-zA-Z]+","",c).replace("{","").replace("}","").replace("\\","")
    return re.sub(r"\s+"," ",c).strip()
def parse_table(tbltxt):
    m=re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", tbltxt, re.S)
    rows=[]
    if m:
        for ln in m.group(0).split("\\\\"):
            ln2=re.sub(r"\\(top|mid|bottom)rule|\\hline|\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}","",ln)
            if "&" in ln2:
                cells=[clean(c) for c in ln2.split("&")]
                if any(cells): rows.append(cells)
    ncol=max((len(r) for r in rows), default=2)
    tbl=[r for r in rows if len(r)==ncol]
    cm=re.search(r"\\caption\{(.+?)\}\s*\\label", tbltxt, re.S)
    cap=cm.group(1).replace("\n"," ").strip() if cm else ""
    return tbl, ncol, cap
def make_table(tbl,ncol):
    w0=0.32; wrest=(1.0-w0)/(ncol-1)
    t=Table(tbl,colWidths=[COLW*w0]+[COLW*wrest]*(ncol-1))
    t.setStyle(TableStyle([
        ("FONT",(0,0),(-1,-1),"Times-Roman",7.3),("FONT",(0,0),(-1,0),"Times-Bold",7.3),
        ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor(CHARCOAL)),("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor(INK)),
        ("LINEABOVE",(0,0),(-1,0),0.8,colors.HexColor(CHARCOAL)),("LINEBELOW",(0,0),(-1,0),0.5,colors.HexColor(CHARCOAL)),
        ("LINEBELOW",(0,-1),(-1,-1),0.8,colors.HexColor(CHARCOAL)),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),1.8),("BOTTOMPADDING",(0,0),(-1,-1),1.8)]))
    return t

# ---------- 6. build story + assemble ----------
def _smallcaps(s,base):
    # classical faux small-caps: initial letters full size, following letters uppercased at ~0.78 pt
    small=round(base*0.78,1)
    def word(w):
        if not w: return w
        out=[]; first=True
        for ch in w:
            if ch.isalpha() and not first:
                out.append(f"<font size={small}>{ch.upper()}</font>")
            else:
                out.append(ch.upper() if ch.isalpha() else ch)
            if ch.isalpha(): first=False
        return "".join(out)
    return " ".join(word(w) for w in s.split(" "))
def _sec_rule(before=10):
    # no rule — the section gap rides on h1_st.spaceBefore, which ReportLab
    # suppresses at a column top, so headings align flush across both columns.
    return Spacer(1,0)
def build_story():
    global _ctr
    _ctr={k:0 for k in THMNAME}; fno=0; st=[]; _tblctr=[0]; _h1seen=[0]
    for b in blocks:
        typ=b[0]
        if typ=="h1":
            # gap rides on the rule's spaceBefore, which ReportLab suppresses at a column top.
            # (The 2nd section formerly force-broke to column 2's top; with the longer
            # problem-first introduction it now flows naturally so The Certificate follows
            # the Introduction directly on page 1.)
            _h1seen[0]+=1
            st+=[_sec_rule(before=10),Paragraph(_smallcaps(inline(b[1]),h1_st.fontSize),h1_st)]
        elif typ=="h2": st.append(Paragraph(_smallcaps(inline(b[1]),h2_st.fontSize),h2_st))
        elif typ=="h3": st.append(Paragraph(inline(b[1]),h3_st))
        elif typ=="para":
            if "@@EQ" in b[1]:
                for p in re.split(r"(@@EQ\d+@@)",b[1]):
                    me=re.match(r"@@EQ(\d+)@@",p)
                    if me: st+=eq_flowable(int(me.group(1)),COLW)
                    elif p.strip(): st.append(Paragraph(inline(p.strip()),body_st))
            else: st.append(Paragraph(inline(b[1]),body_st))
        elif typ=="eq": st+=eq_flowable(b[1],COLW)
        elif typ=="thm": st+=[KeepTogether(thm_flowable(b[1],b[2],b[3]))]
        elif typ=="proof": st+=proof_flowable(b[1])
        elif typ=="fig": fno+=1; st+=fig_flowable(b[1],b[2],COLW,fno)
        elif typ=="table_alpha":
            _tbl,_ncol,_cap=parse_table(b[1]); _tno=_tblctr[0]+1; _tblctr[0]=_tno
            st+=[Spacer(1,3),make_table(_tbl,_ncol),Paragraph(f"<b>Table {_tno}.</b> "+inline(_cap),cap_st)]
        elif typ=="list":
            for it in b[2]: st.append(Paragraph(("• " if b[1]=="itemize" else "")+inline(it),list_st))
        elif typ=="bib":
            for k,r in enumerate(b[1],1): st.append(Paragraph(f"[{k}] "+inline(r),bib_st))
    return st,fno
def header_footer(canvas,doc):
    canvas.saveState()
    # parchment ground across the whole page
    canvas.setFillColor(colors.HexColor(PARCH)); canvas.rect(0,0,PW,PH,fill=1,stroke=0)
    # running text only — no footer rule
    canvas.setFont("Times-Italic",7); canvas.setFillColor(colors.HexColor(WARMGRAY))
    canvas.drawString(MARGIN,10*mm,"The Resolution Calibration Principle")
    canvas.drawRightString(PW-MARGIN,10*mm,"Preprint")
    canvas.setFont("Times-Roman",7.5); canvas.setFillColor(colors.HexColor(CHARCOAL))
    canvas.drawCentredString(PW/2,10*mm,str(doc.page)); canvas.restoreState()
paper_title="The Resolution Calibration Principle: A Certified Bandwidth for Kernel Fields"
_banW=PW-2*MARGIN
# meta-title glyph sizing: scale sigma* png to a target cap height
_msz=PILImage.open("meta_sigma.png").size; _META_H=13*mm; _META_W=_msz[0]*(_META_H/_msz[1])
def _meta_glyph():
    im=Image("meta_sigma.png",width=_META_W,height=_META_H); im.hAlign="CENTER"; return im
def _bannerlist():
    _abs_rule_t=Spacer(1,7)
    _abs_rule_b=Spacer(1,5)
    return [Spacer(1,2),_meta_glyph(),Spacer(1,4),
            Paragraph(paper_title,title_st),
            Paragraph("R. Negulescu<super>1</super>&nbsp;&nbsp;&nbsp;C. Bereanu<super>2</super>",auth_st),
            Paragraph("<super>1</super>The Informational Buildup Foundation (IBF) · <font face='Courier'>radu@ibf.ro</font>",aff_st),
            Paragraph("<super>2</super>The Simion Stoilow Institute of Mathematics of the Romanian Academy (IMAR) · <font face='Courier'>c.bereanu@imar.ro</font>",aff_st),
            _abs_rule_t,Paragraph("Abstract",abs_head),
            Paragraph(inline(re.sub(r"\s+"," ",abstract).strip()),abs_st),_abs_rule_b]
# measure actual banner content height and reserve exactly that (+ small pad), killing dead space
_bh=0.0
for _fl in _bannerlist():
    _w,_h=_fl.wrap(_banW,PH)
    _bh+=_h+_fl.getSpaceBefore()+_fl.getSpaceAfter()
banner_h=min(PH-2*MARGIN-40*mm, _bh+5*mm)
f_ban=Frame(MARGIN,PH-MARGIN-banner_h,PW-2*MARGIN,banner_h,id="ban",showBoundary=0,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
col_y=MARGIN+4*mm; col_h1=PH-MARGIN-banner_h-col_y+5*mm
f_l1=Frame(MARGIN,col_y,COLW,col_h1,id="l1",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
f_r1=Frame(MARGIN+COLW+COLGAP,col_y,COLW,col_h1,id="r1",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
colH=PH-2*MARGIN-6*mm
f_lC=Frame(MARGIN,col_y,COLW,colH,id="lC",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
f_rC=Frame(MARGIN+COLW+COLGAP,col_y,COLW,colH,id="rC",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
story,fno=build_story()
banner=_bannerlist()
doc=BaseDocTemplate("rcp_paper.pdf",pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,topMargin=MARGIN,bottomMargin=MARGIN)
doc.addPageTemplates([PageTemplate(id="first",frames=[f_ban,f_l1,f_r1],onPage=header_footer),
                      PageTemplate(id="rest",frames=[f_lC,f_rC],onPage=header_footer)])
doc.build(banner+[FrameBreak(),NextPageTemplate("rest")]+story)
bts=open("rcp_paper.pdf","rb").read()
pdf=pdfium.PdfDocument("rcp_paper.pdf"); npg=len(pdf)
for idx in range(npg): pdf[idx].render(scale=1.5).to_pil().save(f"pg{idx}.png")
pdf.close()
print("pages:",npg,"bytes:",len(bts),"figs:",fno,"sha:",hashlib.sha256(bts).hexdigest()[:16])

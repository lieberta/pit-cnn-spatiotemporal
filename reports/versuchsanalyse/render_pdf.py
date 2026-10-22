"""Render the report's Markdown tables and formulas as a bookmarked PDF."""
import sys,re,html
from pathlib import Path
sys.path.insert(0,'/tmp/repo-summary-pdf')
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
OUT=Path('reports/versuchsanalyse')
for name,file in [('DV','DejaVuSans.ttf'),('DVB','DejaVuSans-Bold.ttf'),('DVI','DejaVuSans-Oblique.ttf'),('DVM','DejaVuSansMono.ttf')]:pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/dejavu/'+file))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DVB',italic='DVI',boldItalic='DVB')
navy=colors.HexColor('#173e54');teal=colors.HexColor('#087f8c')
style={
 'body':ParagraphStyle('body',fontName='DV',fontSize=9.2,leading=13.4,spaceAfter=7),
 'h1':ParagraphStyle('h1',fontName='DVB',fontSize=23,leading=28,textColor=navy,spaceAfter=14,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='DVB',fontSize=14,leading=18,textColor=navy,spaceBefore=16,spaceAfter=9,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='DVB',fontSize=11,leading=15,textColor=teal,spaceBefore=11,spaceAfter=7,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='DV',fontSize=7.2,leading=10,spaceAfter=0),
 'head':ParagraphStyle('head',fontName='DVB',fontSize=7.2,leading=10,textColor=colors.white),
 'math':ParagraphStyle('math',fontName='DV',fontSize=9.0,leading=14,spaceAfter=0),
}
def fmt(t):
 t=html.escape(t)
 t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t)
 t=re.sub(r'`([^`]+)`',r'<font name="DVM" size="8">\1</font>',t)
 return t
class Doc(SimpleDocTemplate):
 def afterFlowable(self,flow):
  if isinstance(flow,Paragraph) and flow.style.name in ('h1','h2','h3'):
   label=flow.getPlainText();key='h'+str(self.seq.nextf('heading'));self.canv.bookmarkPage(key)
   # Flat section bookmarks avoid illegal jumps when a subheading precedes a section.
   self.canv.addOutlineEntry(label,key,level=0,closed=False)
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#ccd9df'));c.line(43,35,552,35);c.setFillColor(colors.HexColor('#527080'));c.setFont('DV',7)
 c.drawString(43,23,'Physics-enhanced CNN · Mathematische Versuchsanalyse');c.drawRightString(552,23,str(d.page))
lines=(OUT/'Detaillierte_Versuchsanalyse.md').read_text().splitlines();story=[];i=0
while i<len(lines):
 line=lines[i]
 if not line.strip():i+=1;continue
 if line.startswith('```'):
  vals=[];i+=1
  while i<len(lines) and not lines[i].startswith('```'):vals.append(lines[i]);i+=1
  box=Table([[Paragraph('<br/>'.join(fmt(s) for s in vals),style['math'])]],colWidths=[509]);box.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#edf5f7')),('BOX',(0,0),(-1,-1),.4,colors.HexColor('#cbdde5')),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]));story.extend([box,Spacer(1,10)]);i+=1;continue
 if line.startswith('|'):
  vals=[]
  while i<len(lines) and lines[i].startswith('|'):
   cells=[s.strip() for s in lines[i].strip('|').split('|')]
   if not all(re.fullmatch('[-: ]+',s) for s in cells):vals.append(cells)
   i+=1
  n=len(vals[0])
  if vals[0][0]=='Run-ID':widths=[237,45,75,76,76]
  elif n==7:widths=[145,25,48,64,65,65,97]
  elif n==6:widths=[167,27,51,88,88,88]
  elif n==4:widths=[114,142,112,141]
  elif n==5:widths=[115,67,135,96,96] if 'Serie'==vals[0][0] else [109,42,113,113,132]
  else:widths=[509/n]*n
  data=[[Paragraph(fmt(v),style['head' if j==0 else 'cell']) for v in row] for j,row in enumerate(vals)]
  tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f7'),colors.white]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,-1),(-1,-1),.35,colors.HexColor('#ccd9df'))]));story.extend([tab,Spacer(1,10)]);continue
 if line.startswith('#'):
  level=len(line)-len(line.lstrip('#'));story.append(Paragraph(fmt(line[level:].strip()),style['h'+str(min(level,3))]));i+=1;continue
 vals=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')):
  if re.match(r'^(\d+\. |\- )',lines[i]):break
  vals.append(lines[i]);i+=1
 story.append(Paragraph(fmt(' '.join(vals)),style['body']))
pdf=OUT/'Detaillierte_Versuchsanalyse.pdf'
Doc(str(pdf),pagesize=(595.28,841.89),leftMargin=43,rightMargin=43,topMargin=42,bottomMargin=49,title='Mathematische Analyse des Versuchsverlaufs – Physics-enhanced CNN',author='').build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(str(pdf));print('PDF',len(r.pages),'pages',pdf.stat().st_size,'bytes')
for j,p in enumerate(r.pages):
 t=p.extract_text();print(j+1,len(t),t[:75].replace('\n',' '), '| END',t[-55:].replace('\n',' '))

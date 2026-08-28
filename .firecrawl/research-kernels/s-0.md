

Rational Chebyshev Approximations for  the
Error Function*
By W. J. Cody
Abstract. This note presents nearly-best rational approximations ftir  the functions
erf  (i)  and  erfc  (x),  with maximal relative errors ranging down to  between 6  X  10-19 and
3  X  10-20.
In  [1] Hart, et  al.,  present rational approximations for  the  function
erfc  Ox) =  1 -  erf  (.c) =  -%-  /   e~&#39;&#39;\u
Vir J x
valid for 0  ^  x  :£ a,  where a  =  4, 8,  10, or 20. They carefully point out  [1, p.  138]
that these approximations are  not  useful for  computing the  error function
2    fx  -  ■
erf  ix)  =  1 — erfc  ix)  =  —j-  I   e  &#39;
V&#39; ■&#39;o
ill
for  small x  because of  subtraction error, but  they do  not  provide any  alternative.
Hastings&#39; [2] approximations for  erf  (x)  are  no  better, since they explicitly use  the
constant 1 as  an  additive term and  are  chosen to  nearly minimize the  maximum
absolute error rather than the  relative error. Clenshaw&#39;s [3]  Chebyshev series ex-
pansions for  erf  Ox)/x come close to  minimizing relative error, but  his  approximations
are  somewhat inefficient because of  his  choice of  interval and his  restriction to
polynomials.
For  a computer subroutine with entries for  both erf  (x)  and  erfc  (x),  cancellation
error can  be  avoided by  evaluating erf  (.r)  directly and erfc (x)  indirectly (as
1  — erf  (x))  when erf  (x) is smaller in  magnitude than erfc  (x),  and  erf  (x)  indirectly
and  erfc  (a;) directly, otherwise. The changeover point occurs for  |j;|  ~  .47.
In  this note we  present nearly-best rational approximations for  the  functions
erf  (.)•) and  erfc  (x)  with maximal relative errors ranging dowm to  between 6  X  10-19
and 3  X  10~2&quot;. The approximation forms and  intervals used are
erf  (x) ~  xRim0x) ,        \x\  g  .5 ,
erfc  ix) ~  e~x*RiAx) ,        .46875 ^  x g  4.0 ,
; +  ~  RiAl/x2) ¡
Art
where the  Rim(z) are  rational functions of  degree /  in  the  numerator and  to  in  the
denominator. The  relations erf  ( — x)  =  —erf (.r) and  erfc  ( — .r)  =  2  — erfc  (x)  can
be  used to  evaluate the  functions for  negative arguments.
Received .January 24,  1969.
* Work performed under the  auspices of  the  U.  S.  Atomic Energy Commission.
631

632W.  J.  CODY
Table I.  g,m ■ -100 log10
max
fix) -  fÄm(x)
71*7
f(x) = erf(x),               |x|  &lt;  .5
7** «.»»A**************************************** ************
m^
8
** »&gt;!» ******************* » ***************** *****************
0
1
2
3
4
5
139
417
558
800
313     496 688     887 1092
556*   753 960   1172 1390
702     986* 1212   1438 1666
956   1307 1465* 1698 1935
962   1108   1466   1626   1950*
1158   1338   1751   1932
***********************************************************
f (x)  = erfc(x) »            .46875 &lt;x&lt;4.0
************************************************* ** ********
0
1
2
3
4
5
6
7
8
****
«*««
0
1
2
3
4
5
***«
61      109      161      214      270
164      222*   280     340      401      462
376   441     506     572     638
440     597     666     736      806
5Ö2     666     824     897*
1056   1132*
1292   1371
1532   1613*
1775   1859*
*******************************************************
f(x)  = erfc(x),             x &gt; 4.0
.*.********#*****#*#*************#***********#**********
628     756     876
688*   828     958   1081   1198
855     998* 1131   1256
992   1151 1287* 1415
1116   1283 1431   1561*
1232   1405 1558             1824*
a******************************************************
Coefficients for  these approximations only  are  given in  Tables II-IV.

RATIONAL  CHEBYSHEV  APPROXIMATIONS
633
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
#
*
*
*
*
*
*
*
*
*
*
*
*
*
*-
#
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
-*.
oo
oo
#*oo
o  o  o
CM »-»«HI O
o  oo  o
CO CO &lt;M r*  O
OOOOO
CO O
cr o
tn O
-*• o
00 o
lí\  o
CM O
•    •
CO  r-*
moo
Heno
sj- co o
NOO
r» r-  o
tf\ in  O
NNO
m  m o
0s  &gt;*• o
oo oo O
•   •   •
»—«    f*—    44—f
Or-lf» O
&lt;M O  00 o
o  «a- r-  o
&lt;o «-to  o
o  m •&gt;*• o
O  -4- O  O
oo o  m  o
r-  m  Is- o
oo o  cr o
oo a-1— o
m «*• cm o
o  o  co o
lílHOO
Hi4inO
•   •   •    •
CM rp  T-A r-\
co m *o o&gt; o
t+&gt; -4- u\ a* &amp;
íM o  o  •* o
cm ir* m m o
cm r-  co c» o
O  CM r-4 CM O
f-  i*» -d&quot; i** o
«-» co &lt;4- »*■ o
O* Is» -M- *•  o
co r-  -4* co o
•a- o  co cm o
ro  cm Is- cj» o
00  IA CO O  O
P.i-(*(MO
CM O  &lt;M »-i O
vfCMOOO
^  oo &lt;Jh -J3 o
00  CM -4- CO O
•   »  •   •  •
CM r-4 CM CM ü-l
O  CM
O  O
I
*
*
*
*
*
*
*
*
*
Q_   #
*
*
*
*
*
-«-
-» :
—*-
G      *    r-l
HOH
O  »  O
I
CM «-« O  CM
O  O  O  O
I
(CiiMNOH
OOOOO
«o
oo
Is-
o
t-
vO
•    •
co er&gt;
i
O
e&#39;-
er
r-
00  CT» 00
r-  co uv
cm r» o
cv r*- ce
m  m  cy
co r— cm
m  cm m
oo «m o
CO CM &gt;o
r-i  r—  r~»
CM
CO
ir\ cm m  in
Mninco
»»I rH  CO CO
co -a- »~« m
lAOO-H
O  CM 17-4 CO
CO 00  «O »-,
NHCDO
in-ocoh
in  f«  si* ro
f— CM 00  00
vO o&gt;  CO O
vOhvOQ
CM O  &amp;  O
&lt;|- thi er- m
•  •  •  •
CM CM \Q  CO
I
cm (v in  r-  O
o  m a* •* ro
in  r-t  -4- 0s  f—
r&gt;- o  in  cr  cm
-^ cm m m in
er« o  &gt;—• in  t-4
o  cm o  o  co
«a- o  m m o
co m o  O  -o
co m  *-* Is- se
«-» oo m  oo oo
OvO rico rt
oo r~ ^4- -4- o
infótnNO
r-  cm «-» co r-
r~  in  &gt;*• cm Is-
co oo o  t-t  Is-
O» ^   C0 H  Is&quot;
o  r~ co »o m
IM  Is-  r-l  1-1 OO
•   •   •   «   •
CO CO *—* co  **■*
O  *•**   O  «-» nj    o  «-» &lt;M co    O  v* CM. f0  si&quot;
CVJ
ro

W.   J.   CODY
* *   •-« O O  «H r-l  O   O i-l  r-4 «-t i-l  O  O
* *oo     ooooo     oooooo
* *    I
* *ww «,     w    4«,    m     M wwwww~.
* *
* *
* *
* *
* *
* *
* *
* *
* *
m *
* *
* * O
* 41 •«-» (M CO r-4  00  O
* * (Mi\0&gt;OHO
* * œinoor-voo
* *
* •&gt;-* ODOO^CQ        Of-COCOCOO
* cr  „, o  o&gt; o  -t  O      -4- oo »o m  o  O
* * c&gt;HiriCf o       r~-vOcoh-cMO
* * 430CO er  f-  o       mocMincMO
* * oocv&gt;oo      eor-ocr-coo
* *
* * socr&gt;m,-io      o» in  cm r— -4- o
* ¥HO     (J«4lfnMO     CO rO  CO 00  CO O
* *i-40   co co o  &lt;4&quot; O    0*- ro  Is- oo co o
* *  CM O      P-  rH  r-  m  O      »  O  (M rM O   O
Di *   &lt;00 COinCMCOC CMi-40-OinO
41 *      4.     • ••••• •     •••*•
* *  so  --•      t~- *-t  r-*  ir\  T-4      rMinincMr—»h
* *
* 4.——        —  .-• —  —•-,.»,,-,«
4i 4-  &lt;~» CM        OOO-Hin        r-l  «~t r-« O  •■&quot;« «O
4t *oo       OOOOO       OOOOOO
41 *     I      I II II
* 4.WW .-     w     _     w     ~ ww~.~-ww
* 4i
* *
4i *
4&gt; *
4&gt; *
* *
* 4¡
4«            •&gt;-*
* O-  4&gt;
4&gt; 4i
4i *
* * crin -O co  h  oo
* * inr—tMcrcoco
4i * O  O  0s  00  CO sO
4&gt; *
4&gt; * «O  (T  N  H  m         H  4i)  yj  O  í   O«
* * h-í «ovo      mincer— ooin
4( 4; HCOrfl C0 4-         CO CMn O  O  CT&gt;
* 41 co-s-co-&lt;r&gt;-       im«CCOHO-«
4i * cu co  cr&gt; o  co      ir^tTü *in
* 41
41 * COH(MJ&gt;N        O  p-  00  CM —« r~.
4t »  m  r-       co o  r-  o  r-       oo^j-^r-p-oo
* *  co f-       nifn-iHCO      cr» cr* p-  r-  co m
* *  O  co      r—&gt;oror04r-4i      co  oío &gt;o &lt;í  co
4t »coco      roooovoco      n  -o  -í  (\i  &gt;o o
4i *     •     • ••••• ••••••
4» *r-cM      r-NOroin-sj-      (MCM4p-i»j-m«o
* *        i l
41_*_
4t *
4i    &quot;°    *  o  *-•     o»-HCMco&gt;r     o»-&lt;im m4-in
«-*-

RATIONAL  CHEBYSHEV  APPROXIMATIONS
CMCMCMCMCM«~l»-«0
OOOOOOOO
«i  N  h  m  &gt;r m  h  o
ffir&gt;H40fO(MnO
0&gt;  CM CM *Q &lt;t  CM CO O
NÛ-Ort&lt;ÛNNO
cocoo»HP-&gt;a&quot;&quot;4-o
000sOC0000n&lt;7*O
CPOOO-OCT-CMCMO
«or-oincocMvOO
mcMin-o^í-ino^o
o«coco&gt;a-p-co*-to
O  IO &gt;í  &gt;t  -Í  o&gt;ci c
vO&lt;M(7*&gt;0 4&#39;(Mls-0
CMO^OCM^-mCMO
OO-fOlfiHP-O
inmincoooocMO
■t  (7&gt;r&lt;. (Mo O  co  O
oo«™&lt;aop~OP~o
oc^rocOp-P-CMO
COr*-t7^%OOv)P»-4f-447«4
CM CM (M tM r~4 O  r-4  P~
OOOOOOOO
I   I
incMOO&gt;oo&gt;»-4p-
o&lt;MP-cocom«-»vo
O4-»0li.&#39;ûNO
•vOCT*»4Ö&#39;4&quot;fOCOOsP-
t-icgmop-crmiO
•■ors--4-st-&gt;oop---«
.-&lt; oo ro  0s  m  ro  o  r-
oh&lt;î4ûo(ocon
CM»-»CO-4&#39;CMCOf»-00
Of-P-OCMC&gt;M-CO
rlfn&lt;0 1f.NiriNP-
M3inr-ioop-cM«-»m
cm er» oo cm cm oo m  co
t&gt;  co o  CN »o m  4
IftHNCßN^O^sO
•4-0&gt;COCr&gt;0»-««-4lOO
0«-4CT&gt;CM»-|r-4^vO
omcotncocMoco
m4-mH&gt;j-MnH
l
©«-»CMCO-^inOP-
COCOCOCOCOCMCMr-IO
OOOOOOOOO
(&lt;1 «O O  »  4  (Mfl (Cl o
^■OCMP-r-OOmo
o«ooOsOcomcocMO
CMCOtnCMOOP-OP-O
■4&quot;&gt;0»-t»0&gt;!-4tnOs&#39;4-0
Os«-4P-OsOCO&lt;4&quot;COO
4N4lñ(MJ&#39;Nfl0O
P-r-iM.4-00.-»0&gt;0
COCOCOCOOOCOOO
0-4--4&quot;cO&gt;0&lt;M«-«P-0
ujHHMnoaoo
440in4cocoHO
in  p-  0s  co  Is-  47-41 o   &lt;-4 o
coOOCMinom&gt;oo
0&gt;MMr(ÍHOiMO
(OOH^coiHfno^o
coro^r—roooc&gt;-4-o
0tMM0^H^40
COCO&gt;00S&lt;MP-P-P-0
CMM&quot;COCMvOrOi-4inO
«-4 co  &lt;4- co  .-« m  »-* --i  rM
rOrococMCM—«o«-4co
OOOOOOOOO
I   I
CMCM-4-t-ICMtnOOOrO
h-fOi-IH«lNi-ICD4
pjinm4r-iNi-iHfO
U1«0 00  C  H  ^   &lt;}• (J&gt; «O
N4li\0&gt;(r)0 0&gt;CO&lt;r
r— 1-4 o  o  —« cm in  o  co
cr&lt;p»p~cr-ooP-oco
ooovOOt—«cor—o
»■&gt;o-4-P--4-&gt;í-0O^3~a&#39;
ocMcOs-«r~&gt;-4oooo&gt;oc
p~eo&gt;ONa&quot;0&gt;p-cocop-
srr—cMCMr-ico^o^-T
inNH^sOsO^vOIri
cnco&lt;jcMcoor~-cr&gt;co
O&gt;0UP-CMr4C&gt;U»^-in
cop-«4-cMint~tM-oo«-4
COOOinCOCT&gt;4r-!COi-4
OHNC&gt;40f«i(&lt;1Hrri
m   m  H  H  CO  r^  CO  «t  LO.
«Mor^cocrocoOr-i
»~4CMt-400CM«O00inCM
Qc-ICMCO^in   sO t-00
r»
00

W.   J.   CODY
* *
* m— — &lt;-.—..-» Ä*»^— ÄÄÄÄÄ MM«.**_M
» **-40 «-4 »H O         NHOO        CM r-4 O  O  O         CO CM «-4 O  O  O
4i *  o  o OOO   OOOO   OOOOO   OOOOOO
4i 4i  I II             II                 II                     III
41 41    —   w _    «-   V            www  —            .».,».  w   w.   w            **«««»  w w
* *
* *
* *
* * CO 00  00  O* CM O
* 4i &lt;4-r-44&quot;Ors-o
4141 4r4NNOO
* * eOrlN^O       in  HNMMO
* 41 H40inO      »0&gt;H440
* 41
41 * (Ji  00NNO       HH4-ONO
4&lt; 41 ONcoo     r-C&#39;como     o* co oo o  cm o
* * P-COr-IO       O  CM O  CO O        o  «-4 cm -4- co  o
* 4- P» &gt;T «~t O       -4-00p-^4O       co &gt;a- .4- co  o* o
* t-&gt;* in-4&quot;0       Ni-INO       COP-ONO       &gt;0 4  H  CM 03 O
* * 4-i&gt;o in  «O ro  O CM O  O  «-4 O CM CM m  O» CM O
41 41 m  o  © 4  01 00 in-^Ncoo &gt;o h  o  o&gt; cm o
4. *ho ö  «-» o      co »-» r-  ©     o»0O4r-4o     r-  co cm «4- c« o
4t *   sOO P-HO         Sp  «4&quot; CM ©         CO CM r-4 O  ©          O  «-• O  00  .-,  ©
41 4&gt;00 040       «04NO       CMCMnCMO       44rt(MOO
* *
* »P-O CMCMO OfMrtO (MO P-  fO  o ocoinincMO
4&lt; 411-40 4ino     cm m f*- ©     o  o  o  co o       cm oo o  o  tn o
41 m  Oí  O cT-ÎO        OinNO        CM CO r-4 Is-  O         mrtlTiNeOO
* «OO O  «-4 O        OO^COO       vOHincOO       (OinNr-*00
* 4&lt;-4-0 incMO       C000-4-O       OCPOOO       CO O  CM 00  m  O
41 * .   . •••    ••••    •••••    ••••••
4&gt; *   S?* «H ,»» ÇT&lt; r-4        nÍ&quot; *4&quot; »-4 «r-4        r-4  p-4 «-4 «-I »-I        CM «O m  r-4  CM r-4
* *
*-*-
* *
41 4-
4t *t-4CM CM »-4 CM        tM4i-4«-4CM        CO CM «t-4 r-l  CM -41 CM r-t  .-4  «H  CM
41 *oo OOO   OOOO   OOOOO OOOOOO
4&gt; 4«    I     I III            I     I     I     I            I     I     I     I     I I     I     I     I     I     I
* *      «««•   «W W*-W               *47&gt;«444&#39;   «*     —                 W     —     W     W     .», W     W     ^      W     —     &quot;-*
41 4«
* 41
41 *
* 41
mm p-  co -a- cr» m  oo
mm inp-ocvjrocj&gt;
m.« r-4 CM CM ■* O  «a*
m &lt;-&gt;T^4&lt; ^  «fooovo fn«o&gt;oiT4co
m 4&lt; (»-cofoooooovo»^^^^
41 *
m m pip»oin&lt;o ooMM4mo*
m * oo o  o  o      cMo»or-M- r-  cm &lt;r&gt; &#39;sp cm ©
m 41 Is- sP cm sp      &gt;a- m  co -4- oo co cm cm ©  co cm
m 41 iTH-ON        m(M&gt;OvOH 00  &gt;$■ CM 00  CM O
m * co o  m     o&gt;cooop~     co ro  oi  o&gt; &lt;j- a» p-- *-4 0s ««4 ro
m m
m 4« in  cm oc      co o  co r-       ocmcooco cm co .-4 -^  o  Is-
m 4« mr^-f-i      co cm M&quot; cm      r-  o  in  o  p- m  ^a- »-4 cr  ct&gt; co
41 *^i&quot;&gt;4- CO CO CM        MJinO^O^         f-  O  CO CO 0&gt; r-4  r-4 -»O C&gt; &gt;*&quot; r-4
m *  &lt;■ o «a-t*-&gt;o      p-incM»-«      oho-oia vo in  cm cr&gt; ro  r-
m m m  ro \o  ir» cm      cm cr&gt; ©  m      r-  cr&gt; m  ro  &gt;4- h  co p» co &gt;o co
m m
* *   OOC CT*  00  CM         CO CO r-l  &lt;r&gt;         ©  ©  sO r-4  CM O1 f-  H  4   &gt;û  Cl
m mvO«~» croco     oo»-««-*     «-» co m  «o cr- -»r co ao M&quot; cm m
m mrocM p» o  co      cocrcrco      o  r*- a» o  »-» t»- ce p-  co co r-4
m rn-4-oo invOvO      «-icrco^a-      o» -a- o  co co co ©  m  ©  m  co
m *    CM O CM CT&gt; r-4        CM «-4 -J-  CM         O»- O» CM F— CM lAOCMOOO
m m*« •••       ••*•       •*•••       ••••••
m m «-4 cr -4- «-4 in      «h^cvco      in  *a- cm cm cm nO HHmmn
* «   l    1 ill lili        1    1    1   1    1 1    1   1    1    1   l
41-41-
m m
41 *r°4lO«-« O   &quot;-»&lt;M          ©  4-4  CM  CO          0&lt;-4CMCO&gt;a- O   H  OJ  Cl   4  Ul
4«-4i-
* C     *   »-4 CM CO                             &gt;í «n

RATIONAL   CHEBYSHEV  APPROXIMATIONS
637
Table I presents the  initial segments of the  Lx Walsh arrays while Tables II,  III,
and  IV  present selected approximations. All  approximations were generated using a
standard version of  the  Remes algorithm [4] on  a  CDC 3600. The master function
routines used continued-fraction expansions described in  [1] and  were verified to  be
accurate to  at  least 22S. Finally, the  accuracy of  the  approximations as  presented
here was  verified by  comparison against the  master routines using 5000 pseudo-
random arguments.
Argonne National Laboratory
Argonne, Illinois 60439
1.  J.  F.  Hart, et  al.,  Computer Approximations, Siam Series in  Appl. Math., Wiley, New
York, 1968.
2.  C.  Hastings, Jr.,  Approximations for  Digital Computers, Princeton Univ. Press, Princeton,
N. J.,  1955. MR  16, 963.
3.  C.  W.  Clenshaw, Chebyshev Series for  Mathematical Functions, National Physical Lab.
Math. Tables, vol. 5, H.M.S.O., London, 1962. MR 26 #362.
4.  W.  J.  Cody, W.  Fraser &amp; J.  F.  Hart, &quot;Rational Cheb3&#39;shev approximations using linear
equations,&quot; Numer. Math., v.  12, 1968, pp.  242-251.
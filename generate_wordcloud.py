# Find the words taken for this plot in the notebook Top_Words_FFM_RENT.ipynb
from functions import generate_wordcloud

active = 1
if active == 1:
    text_cat0 = "wohnheim,studentenwohnheim,einschreiben,lage wohnheim,hochschule,immatrikuliert,raunheim,wohnheims,zimmer studentenwohnheim,hessisch hochschule,hessisch,eingeschrieben,hochbett,innen,wohnheim liegen,bonames mitte,campus,hochschule einschreiben,bonames,studentenwohnheim handeln,fahren meist,wohnheim befinden,direkt campus,hornbach,studentenwerk,befinden studentenwohnheim,eingeschrieben student,meist min,verkehrstechnisch super,partykeller,studentenwohnheims,infos www,musst hessisch,studentenverbindung,studierend,ben,holzhausenstrasse miquel,banane,wohnung,soziale arbeit,katholisch,ev,eisdiele post,studieren,klettenberg,ackermann,gwh,soziale,ackermann de,wg ackermann,wg studentenwohnheim,takt einkaufsmoeglichkeiten,campus westend,mitbewohner innen,fernstudenten,fernstudenten praktikanten,gasthoerer,gasthoerer fernstudenten,anschliessen bahn,hochschule immatrikuliert,flughafen fahren,frankfurt frankfurter,ikea hornbach,ben gurion,gurion,gurion ring,zimmer wohnheim,min takt,giessener,studentenwohnheim liegen,flughafen anschliessen,kalbach,super frankfurt,wg eigentlich,frei bett,rassismus,baecker eisdiele,flatrate ebenfalls,kommunikationstechnisch,studentenwerks,susanna klettenberg,raunheim verkehrstechnisch,interesse reine,fitnessraum,bzw fachhochschule,fachhochschule hauptsitz,hauptsitz hessen,ev studentenwohnheime,studentenwohnheime de,www ev,hoechst,rené,sexismus,table,giessener strasse"
    text_cat4 ="smart tv,community,smart,co living,stilvoll,warmwasser sonstig,apartment,besonderheit,sonstig umlagen,via phone,provide,pauschalmiete,zoll smart,beruecksichtigen handeln,gez warmwasser,service,hochwertig,einrichtung netflix,momente community,waschtrockner kombination,co,parkett versehen,cleaningservice,living,boxspringbett cm,modern,hinaus gehoeren,boxspringbett,handeln pauschalmiete,homefully,weekly,contact,spacious apartment,zuverlaessig putzfrau,business wg,umfeld,spacious,woechentlich zuverlaessig,wohnung woechentlich,flat vorhanden,wmf,fully,anspruechen,hochwertig boxspringbett,wlan flat,heizung gez,kitchen utensils,hochmodern,service room,putzfrau,contracts tenants,english provide,provide flexible,sodass gutes,komplettmoeblierung,anspruechen genuegen,zimmer inmitten,backen weit,besonderheit zusammenleben,bett cleaningservice,broadband connection,co somit,community laesst,effektiv haushalt,haushalt verfuegung,hochmodern zoll,kleiderschrank hochmodern,kombination effektiv,kosten zaehlen,laesst internet,pauschalmiete beruecksichtigen,pfannen set,schoen momente,sicherlich besonderheit,topf pfannen,tv ebenfalls,tv geraeumig,umlagen pauschalmiete,vereinfachen raum,wifi broadband,wmf topf,woche smart,zusammenleben vereinfachen,details contact,know contact,tenants let,erfolgreich,cleaning service,comes appliances,complete furnished,fi weekly,flexible rental,home monthly,kitchen comes,needed feel,offer complete,rental contracts,utilities wi,please scroll,weekly cleaning,gesamtmietpreis,stehen modern,ausdehnen freunden,ausstatten ausdehnen,benoetigen wmf,besonders wg,ofen modern,reinigen besonders,set stilvoll,sowie vollstaendig,stilvoll ausstatten,stilvoll parkett,connection wohnung,bottom english,duerfen umfeld,eigen anspruechen,erfolgreich zeit,garantieren gut,genuegen rahmenbedingungen,gewaehrleisten sorgen,inkl telefonnummer,konstanz positiv,lebensabschnitt stehen,mieter selben,positiv gemeinsam,rahmenbedingungen erfolgreich,schaffen hause,scroll bottom,sorgen konstanz,suchen business,umfeld eigen,umfeld schaffen,wichtig umfeld,rent utilities,vollstaendig kueche,cookware,cookware needed,utensils cookware,komplettmoeblierung zimmer,zaehlen komplettmoeblierung,english besonderheit,main garantieren,neu geraeumig,reinigungskraft please,broadband,young,telefonnummer rufen,besonderheit queensize,cleaningservice woche,cm ausstatten"
    generate_wordcloud(text=text_cat0, name="words_cat0")
    generate_wordcloud(text=text_cat4, name="words_cat4")
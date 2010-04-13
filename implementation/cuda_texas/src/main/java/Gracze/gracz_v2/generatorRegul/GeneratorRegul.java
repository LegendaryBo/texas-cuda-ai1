package Gracze.gracz_v2.generatorRegul;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.RegulaAbstrakcyjnaDobijania;
import reguly.LicytacjaNaWejscie.RegulaBardzoWysokieKarty;
import reguly.LicytacjaNaWejscie.RegulaCzyKolorWRece;
import reguly.LicytacjaNaWejscie.RegulaCzyParaWRece;
import reguly.LicytacjaNaWejscie.RegulaOgraniczenieStawki;
import reguly.LicytacjaNaWejscie.RegulaStalaStawka;
import reguly.LicytacjaNaWejscie.RegulaWymaganychGlosow;
import reguly.LicytacjaNaWejscie.RegulaWysokieKarty;
import reguly.LicytacjaPozniej.RegulaJestRezultat;
import reguly.LicytacjaPozniej.RegulaLicytujGdyMalaStawka;
import reguly.dobijanie.RegulaDobijacZawsze;
import reguly.dobijanie.RegulaDobijajGdyBrakujeX;
import reguly.dobijanie.RegulaDobijajGdyDobraKarta;
import reguly.dobijanie.RegulaDobijajGdyParaWRece;
import reguly.dobijanie.RegulaDobijajGdyWysokaReka;
import Gracze.gracz_v3.GeneratorRegulv3;

/**
 * Metody statyczne, ktore generuja zestawy regul
 * 
 * 
 * GENERATOR TRZEBA NAJPIERW ZAINICJOWAC METODA init()
 * 
 * @author Kacper Gorski
 *
 */
public final class GeneratorRegul {

  public static int rozmiarGenomu = 0;
  
  public static int[] poczatekCzesciCzyGrac = new int[4];
  public static int[] poczatekCzesciStawka = new int[4];
  public static int[] poczatekCzesciDobijanie = new int[4];
  

  
  public static void init() {
    GeneratorRegulv3.init();
    
    ArrayList<RegulaAbstrakcyjna> pListaWszystkichRegul = generujKompletRegul(null);
    rozmiarGenomu = getDlugoscRegul(pListaWszystkichRegul);
    
    poczatekCzesciCzyGrac[0] = 0;
    poczatekCzesciStawka[0] = getDlugoscRegul(generujRegulyNaWejscie(null)) + poczatekCzesciCzyGrac[0];  
    poczatekCzesciDobijanie[0] = getDlugoscRegul(generujRegulyStawkaRunda1(null)) + poczatekCzesciStawka[0];
       
    poczatekCzesciCzyGrac[1] = getDlugoscRegul2(generujRegulyDobijanieRunda1(null)) + poczatekCzesciDobijanie[0];
    poczatekCzesciStawka[1] = getDlugoscRegul(generujRegulyNaWejscieRundyKolejne(null, 1)) + poczatekCzesciCzyGrac[1];
    poczatekCzesciDobijanie[1] = getDlugoscRegul(generujRegulyStawkaRundyKolejne(null, 1)) + poczatekCzesciStawka[1];
    
    poczatekCzesciCzyGrac[2] = getDlugoscRegul2(generujRegulyDobijanieRundyKolejne(null, 2)) + poczatekCzesciDobijanie[1];
    poczatekCzesciStawka[2] = getDlugoscRegul(generujRegulyNaWejscieRundyKolejne(null, 2)) + poczatekCzesciCzyGrac[2];
    poczatekCzesciDobijanie[2] = getDlugoscRegul(generujRegulyStawkaRundyKolejne(null, 2)) + poczatekCzesciStawka[2];
    
    poczatekCzesciCzyGrac[3] = getDlugoscRegul2(generujRegulyDobijanieRundyKolejne(null, 3)) + poczatekCzesciDobijanie[2];
    poczatekCzesciStawka[3] = getDlugoscRegul(generujRegulyNaWejscieRundyKolejne(null, 3)) + poczatekCzesciCzyGrac[3];
    poczatekCzesciDobijanie[3] = getDlugoscRegul(generujRegulyStawkaRundyKolejne(null, 3)) + poczatekCzesciStawka[3];    
  }

  // indeksy wlacznikow genow odpowiedzialnych za wejscie do gry w 1 rundzie
  public static int[] indeksyGenowNaWejscie = null;
  

  
  public static RegulaWymaganychGlosow regulaWymaganychGlosowNaWejscieR1 = null;
  public static RegulaCzyParaWRece regulaCzyParaWReceNaWejscieR1 = null;
  public static RegulaCzyKolorWRece regulaCzyKolorWReceNaWejscieR1 = null;
  public static RegulaOgraniczenieStawki regulaOgraniczenieStawkiNaWejscieR1 = null;
  public static RegulaWysokieKarty regulaWysokieKartyNaWejscieR1 = null;
  public static RegulaBardzoWysokieKarty regulaBardzoWysokieKartyNaWejscieR1 = null;
  
  public static ArrayList<RegulaAbstrakcyjna> generujRegulyNaWejscie(EvBinaryVectorIndividual aIndividual) {
    
    final int ROZMIAR_WYMAGANYCH_GLOSOW = 6; // ile bitow
    final int ROZMIAR_GLOSOW = 4; // ile bitow na glos
    final int ROZMIAR_STAWKI_DLA_REGULY = 10; // ile bitow na glos

    
    int pRozmiar = poczatekCzesciCzyGrac[0];
    
    ArrayList<RegulaAbstrakcyjna> pListaRegul = new ArrayList<RegulaAbstrakcyjna>();
    
    RegulaAbstrakcyjna pRegula = new RegulaWymaganychGlosow(pRozmiar, ROZMIAR_WYMAGANYCH_GLOSOW);
    regulaWymaganychGlosowNaWejscieR1 = (RegulaWymaganychGlosow) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();
    
    if (indeksyGenowNaWejscie == null)
      indeksyGenowNaWejscie = new int[5];
    
    indeksyGenowNaWejscie[0] = pRozmiar;
    
    pRegula = new RegulaCzyParaWRece(pRozmiar, aIndividual, ROZMIAR_GLOSOW);
    regulaCzyParaWReceNaWejscieR1 = (RegulaCzyParaWRece) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();  
    
    indeksyGenowNaWejscie[1] = pRozmiar;
    
    pRegula = new RegulaCzyKolorWRece(pRozmiar, ROZMIAR_GLOSOW);
    regulaCzyKolorWReceNaWejscieR1 = (RegulaCzyKolorWRece) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();      
    
    indeksyGenowNaWejscie[2] = pRozmiar;
    
    pRegula = new RegulaOgraniczenieStawki(pRozmiar, ROZMIAR_GLOSOW, ROZMIAR_STAWKI_DLA_REGULY);
    regulaOgraniczenieStawkiNaWejscieR1 = (RegulaOgraniczenieStawki) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();      
    
    indeksyGenowNaWejscie[3] = pRozmiar;
    
    pRegula = new RegulaWysokieKarty(pRozmiar, ROZMIAR_GLOSOW);
    regulaWysokieKartyNaWejscieR1 = (RegulaWysokieKarty) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();          
    
    indeksyGenowNaWejscie[4] = pRozmiar;
    
    pRegula = new RegulaBardzoWysokieKarty(pRozmiar, ROZMIAR_GLOSOW);
    regulaBardzoWysokieKartyNaWejscieR1 = (RegulaBardzoWysokieKarty) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();           
    
    return pListaRegul;
  }
  
  // indeksy wlacznikow genow odpowiedzialnych za wejscie do gry w 1 rundzie
  public static int[] indeksyGenowStawka1 = null;
  
  public static int stawkaR1ParaWReceDlugosc;
  public static int stawkaR1KolorWReceDlugosc;
  public static int stawkaR1WysokaKartaWReceDlugosc;
  
  public static RegulaCzyParaWRece regulaCzyParaWReceStawkaR1 = null;
  public static RegulaCzyKolorWRece regulaCzyKolorWReceStawkaR1 = null;
  public static RegulaCzyKolorWRece regulaCzyKolorWRece2StawkaR1 = null;
  public static RegulaWysokieKarty regulaWysokieKartyStawkaR1 = null;
  public static RegulaBardzoWysokieKarty regulaBardzoWysokieKartyStawkaR1 = null;
  public static RegulaStalaStawka regulaStalaStawkaStawka1R1 = null;
  public static RegulaStalaStawka regulaStalaStawkaStawka2R1 = null;
  public static RegulaStalaStawka regulaStalaStawkaStawka3R1 = null;
  
  public static ArrayList<RegulaAbstrakcyjna> generujRegulyStawkaRunda1(EvBinaryVectorIndividual aIndividual) {
    
    final int PARA_W_RECE_BITOW = 8; // ile bitow
    final int KOLOR_W_RECE_BITOW = 8; // ile bitow
    final int WYSOKA_KARTA_W_RECE_BITOW = 8; // ile bitow
    
    int pRozmiar = poczatekCzesciStawka[0];
  
    stawkaR1ParaWReceDlugosc = PARA_W_RECE_BITOW;
    stawkaR1KolorWReceDlugosc = KOLOR_W_RECE_BITOW;
    stawkaR1WysokaKartaWReceDlugosc = WYSOKA_KARTA_W_RECE_BITOW;    
    
    ArrayList<RegulaAbstrakcyjna> pListaRegul = new ArrayList<RegulaAbstrakcyjna>();  
  
    if (indeksyGenowStawka1 == null)
      indeksyGenowStawka1 = new int[8];
    
    indeksyGenowStawka1[0] = pRozmiar;    
    
    RegulaAbstrakcyjna pRegula = new RegulaCzyParaWRece(pRozmiar, aIndividual, PARA_W_RECE_BITOW);
    regulaCzyParaWReceStawkaR1 = (RegulaCzyParaWRece) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();  
    
    indeksyGenowStawka1[1] = pRozmiar;  
    
    pRegula = new RegulaCzyKolorWRece(pRozmiar, KOLOR_W_RECE_BITOW);
    regulaCzyKolorWReceStawkaR1 = (RegulaCzyKolorWRece) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();      
    
    indeksyGenowStawka1[2] = pRozmiar;  
    
    pRegula = new RegulaCzyKolorWRece(pRozmiar, KOLOR_W_RECE_BITOW);
    regulaCzyKolorWRece2StawkaR1 = (RegulaCzyKolorWRece) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();          
    
    indeksyGenowStawka1[3] = pRozmiar;  
    
    pRegula = new RegulaWysokieKarty(pRozmiar, WYSOKA_KARTA_W_RECE_BITOW);
    regulaWysokieKartyStawkaR1 = (RegulaWysokieKarty) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();            
    
    indeksyGenowStawka1[4] = pRozmiar;  
    
    pRegula = new RegulaBardzoWysokieKarty(pRozmiar, WYSOKA_KARTA_W_RECE_BITOW);
    regulaBardzoWysokieKartyStawkaR1 = (RegulaBardzoWysokieKarty) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();         
    
    indeksyGenowStawka1[5] = pRozmiar;  
    
    pRegula = new RegulaStalaStawka(pRozmiar, aIndividual, 3);
    regulaStalaStawkaStawka1R1 = (RegulaStalaStawka) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();      
    
    indeksyGenowStawka1[6] = pRozmiar;  
    
    pRegula = new RegulaStalaStawka(pRozmiar, aIndividual, 5);
    regulaStalaStawkaStawka2R1 = (RegulaStalaStawka) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();      
    
    indeksyGenowStawka1[7] = pRozmiar;  
    
    pRegula = new RegulaStalaStawka(pRozmiar, aIndividual, 9);
    regulaStalaStawkaStawka3R1 = (RegulaStalaStawka) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();         
    
    return pListaRegul;
  }
  
  // indeksy wlacznikow genow odpowiedzialnych za wejscie do gry w 1 rundzie
  public static int[] indeksyGenowDobijanie1 = null;  
  
  public static RegulaDobijacZawsze regulaDobijacZawszeDobijanieR1 = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX1R1 = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX2R1 = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX3R1 = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX4R1 = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX5R1 = null;
  
  public static RegulaDobijajGdyParaWRece regulaDobijajGdyParaWReceX1R1 = null;
  public static RegulaDobijajGdyParaWRece regulaDobijajGdyParaWReceX2R1 = null;
  public static RegulaDobijajGdyParaWRece regulaDobijajGdyParaWReceX3R1 = null;
  
  public static RegulaDobijajGdyWysokaReka regulaDobijajGdyWysokaRekaX1R1 = null;
  public static RegulaDobijajGdyWysokaReka regulaDobijajGdyWysokaRekaX2R1 = null;
  public static RegulaDobijajGdyWysokaReka regulaDobijajGdyWysokaRekaX3R1 = null;
  
  public static ArrayList<RegulaAbstrakcyjnaDobijania> generujRegulyDobijanieRunda1(EvBinaryVectorIndividual aIndividual) {
    
    int pRozmiar = poczatekCzesciDobijanie[0];
    
    ArrayList<RegulaAbstrakcyjnaDobijania> pListaRegul = new ArrayList<RegulaAbstrakcyjnaDobijania>();      
    
    if (indeksyGenowDobijanie1 == null)
      indeksyGenowDobijanie1 = new int[13];
    
    indeksyGenowDobijanie1[0] = pRozmiar;        
    
    RegulaAbstrakcyjnaDobijania pRegula = new RegulaDobijacZawsze(pRozmiar);
    regulaDobijacZawszeDobijanieR1 = (RegulaDobijacZawsze) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();      
 
    indeksyGenowDobijanie1[1] = pRozmiar;    
    
    pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar,  0.3);
    regulaDobijajGdyBrakujeX1R1 = (RegulaDobijajGdyBrakujeX) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();     
    
    indeksyGenowDobijanie1[2] = pRozmiar;    
    
    pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar,  0.6);
    regulaDobijajGdyBrakujeX2R1 = (RegulaDobijajGdyBrakujeX) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();        
    
    indeksyGenowDobijanie1[3] = pRozmiar;    
    
    pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar,  0.8);
    regulaDobijajGdyBrakujeX3R1 = (RegulaDobijajGdyBrakujeX) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();          
    
    indeksyGenowDobijanie1[4] = pRozmiar;    
    
    pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar,  0.1);
    regulaDobijajGdyBrakujeX4R1 = (RegulaDobijajGdyBrakujeX) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();          
    
    indeksyGenowDobijanie1[5] = pRozmiar;    
    
    pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar,  0.3);
    regulaDobijajGdyBrakujeX5R1 = (RegulaDobijajGdyBrakujeX) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();       
    
    indeksyGenowDobijanie1[6] = pRozmiar;    

    
    pRegula = new RegulaDobijajGdyParaWRece(pRozmiar,  0.1);
    regulaDobijajGdyParaWReceX1R1 = (RegulaDobijajGdyParaWRece) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();          
    
    indeksyGenowDobijanie1[7] = pRozmiar;    
    
    pRegula = new RegulaDobijajGdyParaWRece(pRozmiar, 0.3);
    regulaDobijajGdyParaWReceX2R1 = (RegulaDobijajGdyParaWRece) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();        
    
    indeksyGenowDobijanie1[8] = pRozmiar;    
    
    pRegula = new RegulaDobijajGdyParaWRece(pRozmiar, 0.6);
    regulaDobijajGdyParaWReceX3R1 = (RegulaDobijajGdyParaWRece) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();         
    
      
    
    indeksyGenowDobijanie1[9] = pRozmiar; 
    
    pRegula = new RegulaDobijajGdyWysokaReka(pRozmiar, 0.1);
    regulaDobijajGdyWysokaRekaX1R1 = (RegulaDobijajGdyWysokaReka) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();        
    
    indeksyGenowDobijanie1[10] = pRozmiar; 
    
    pRegula = new RegulaDobijajGdyWysokaReka(pRozmiar, 0.3);
    regulaDobijajGdyWysokaRekaX2R1 = (RegulaDobijajGdyWysokaReka) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();   
    
    indeksyGenowDobijanie1[11] = pRozmiar; 
    
    pRegula = new RegulaDobijajGdyWysokaReka(pRozmiar, 0.6);
    regulaDobijajGdyWysokaRekaX3R1 = (RegulaDobijajGdyWysokaReka) pRegula;
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();     
    
    indeksyGenowDobijanie1[12] = pRozmiar; 
    
    return pListaRegul;
  }
  
  
  
  
  
  
  
  
  
  
  public static RegulaJestRezultat[][] regulaJestRezultatWejscieRX = new RegulaJestRezultat[3][7];
  public static RegulaOgraniczenieStawki[][] regulaOgraniczenieStawkiWejscieRX = new RegulaOgraniczenieStawki[3][5]; 
  
  public static ArrayList<RegulaAbstrakcyjna> generujRegulyNaWejscieRundyKolejne(EvBinaryVectorIndividual aIndividual,
        int aRunda) {
      
    int pRozmiar = poczatekCzesciCzyGrac[aRunda-1];

    
    ArrayList<RegulaAbstrakcyjna> pListaRegul = new ArrayList<RegulaAbstrakcyjna>();    

  
    final int ROZMIAR_WYMAGANYCH_GLOSOW = 7; // ile bitow
    
    RegulaAbstrakcyjna pRegula = new RegulaWymaganychGlosow(pRozmiar, ROZMIAR_WYMAGANYCH_GLOSOW);
    pListaRegul.add( pRegula );
    pRozmiar += pRegula.getDlugoscReguly();    
    
    
    for (int i=2; i < 9; i++) {
      pRegula = new RegulaJestRezultat(pRozmiar, 5+i, i);
      regulaJestRezultatWejscieRX[aRunda-1][i-2] = (RegulaJestRezultat)pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();         
    }
 
    for (int i=0; i < 5; i++) {
      pRegula = new RegulaOgraniczenieStawki(pRozmiar, 4, 8);
      regulaOgraniczenieStawkiWejscieRX[aRunda-1][i] = (RegulaOgraniczenieStawki)pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();         
    }
    
    return pListaRegul;
    
  }
  
  
  public static RegulaJestRezultat[][] regulaJestRezultatStawkaRX = new RegulaJestRezultat[3][7];
  public static RegulaLicytujGdyMalaStawka[][] regulaLicytujGdyMalaStawkaWejscieRX = new RegulaLicytujGdyMalaStawka[3][5];  
  
  public static ArrayList<RegulaAbstrakcyjna> generujRegulyStawkaRundyKolejne(EvBinaryVectorIndividual aIndividual,
      int aRunda) {
    
    int pRozmiar = poczatekCzesciStawka[aRunda -1];
    
    ArrayList<RegulaAbstrakcyjna> pListaRegul = new ArrayList<RegulaAbstrakcyjna>();    
    
    RegulaAbstrakcyjna pRegula = null;
    
    
    for (int i=2; i < 9; i++) {
      pRegula = new RegulaJestRezultat(pRozmiar, 5+i,i);
      regulaJestRezultatWejscieRX[aRunda-1][i-2] = (RegulaJestRezultat)pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();         
    }
    
    for (int i=0; i < 5; i++) {
      pRegula = new RegulaLicytujGdyMalaStawka(pRozmiar,  4, 8);
      regulaLicytujGdyMalaStawkaWejscieRX[aRunda-1][i] = (RegulaLicytujGdyMalaStawka)pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();         
    }   
    
    
    return pListaRegul;
    
  }

  
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX_1RX = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX_2RX = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX_3RX = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX_4RX = null;
  public static RegulaDobijajGdyBrakujeX regulaDobijajGdyBrakujeX_5RX = null;
  
  public static RegulaDobijajGdyDobraKarta[][] regulaDobijajGdyDobraKartaXRX = new RegulaDobijajGdyDobraKarta[3][7];
  
  public static ArrayList<RegulaAbstrakcyjnaDobijania> generujRegulyDobijanieRundyKolejne(EvBinaryVectorIndividual aIndividual,
          int aRunda) {
      
      int pRozmiar = poczatekCzesciDobijanie[aRunda - 1];
      
      ArrayList<RegulaAbstrakcyjnaDobijania> pListaRegul = new ArrayList<RegulaAbstrakcyjnaDobijania>();      
      
      RegulaAbstrakcyjnaDobijania pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar, 0.3);
      regulaDobijajGdyBrakujeX_1RX = (RegulaDobijajGdyBrakujeX) pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();     
      
      pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar,  0.6);
      regulaDobijajGdyBrakujeX_2RX = (RegulaDobijajGdyBrakujeX) pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();        
      
      pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar,  0.8);
      regulaDobijajGdyBrakujeX_3RX = (RegulaDobijajGdyBrakujeX) pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();          
      
      pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar,  0.1);
      regulaDobijajGdyBrakujeX_4RX = (RegulaDobijajGdyBrakujeX) pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();          
      
      
      pRegula = new RegulaDobijajGdyBrakujeX(pRozmiar, 0.3);
      regulaDobijajGdyBrakujeX_5RX = (RegulaDobijajGdyBrakujeX) pRegula;
      pListaRegul.add( pRegula );
      pRozmiar += pRegula.getDlugoscReguly();       
      
      
      for (int i=2; i < 9; i++) {
        
          pRegula = new RegulaDobijajGdyDobraKarta(pRozmiar, i);
          regulaDobijajGdyDobraKartaXRX[aRunda-1][i-2] = (RegulaDobijajGdyDobraKarta)pRegula;
          pListaRegul.add( pRegula );
          pRozmiar += pRegula.getDlugoscReguly();             
        
      }
      
      
      return pListaRegul;
  }
    
  
  
  
  
  
  // Zwraca komplet regul
  public static ArrayList<RegulaAbstrakcyjna> generujKompletRegul(EvBinaryVectorIndividual aIndividual) {
    
    ArrayList<RegulaAbstrakcyjna> pListaWszystkichRegul = new ArrayList<RegulaAbstrakcyjna>();
    
    ArrayList<RegulaAbstrakcyjna> pListaRegul = generujRegulyNaWejscie(aIndividual);
    int pDlugoscRegul = getDlugoscRegul(pListaRegul);
    pListaWszystkichRegul.addAll( pListaRegul );
    
    pListaRegul = generujRegulyStawkaRunda1(aIndividual);
    pDlugoscRegul += getDlugoscRegul(pListaRegul);
    pListaWszystkichRegul.addAll( pListaRegul );
    
    ArrayList<RegulaAbstrakcyjnaDobijania> pListaRegu2 = generujRegulyDobijanieRunda1(aIndividual);
    pDlugoscRegul += getDlugoscRegul2(pListaRegu2);
    pListaWszystkichRegul.addAll( pListaRegu2 );    
    
    for (int i=1; i <= 3; i++) {
    
      pListaRegul = generujRegulyStawkaRundyKolejne(aIndividual, i);
      pDlugoscRegul += getDlugoscRegul(pListaRegul);
      pListaWszystkichRegul.addAll( pListaRegul );      
      
      pListaRegul = generujRegulyNaWejscieRundyKolejne(aIndividual, i);
      pDlugoscRegul += getDlugoscRegul(pListaRegul);
      pListaWszystkichRegul.addAll( pListaRegul );   
      
      pListaRegu2 = generujRegulyDobijanieRundyKolejne(aIndividual, i);
      pDlugoscRegul += getDlugoscRegul2(pListaRegu2);
      pListaWszystkichRegul.addAll( pListaRegu2 );    
      
    }
    
    return pListaWszystkichRegul;
    
  }
  
  
  
  private static int getDlugoscRegul(ArrayList<RegulaAbstrakcyjna> aReguly) {
    
    int pDlugosc = 0;
    
    for (RegulaAbstrakcyjna pRegula : aReguly) {
      pDlugosc += pRegula.getDlugoscReguly();
    }
    
    return pDlugosc;
  }
  
    /****** pierdzielone debilizmy geneirxow */
  private static int getDlugoscRegul2(ArrayList<RegulaAbstrakcyjnaDobijania> aReguly) {
    
    int pDlugosc = 0;
    
    for (RegulaAbstrakcyjna pRegula : aReguly) {
      pDlugosc += pRegula.getDlugoscReguly();
    }
    
    return pDlugosc;
  }  
  
  
  
  
  
  
  // podaje informacje o rozmiarze czesci genomow odpowiedzialnych za poszczegolne rundy
  public static String getInfo() {
    
    String pString = "Rozmiar czesci genomow odpowiedzialnych za poszczegolne rundy:\n";

    pString += "Runda 1: ";
    pString += (poczatekCzesciCzyGrac[1]-poczatekCzesciCzyGrac[0])+"\n";
    
    pString += "Runda 2: ";
    pString += (poczatekCzesciCzyGrac[2]-poczatekCzesciCzyGrac[1])+"\n";    
    
    pString += "Runda 3: ";
    pString += (poczatekCzesciCzyGrac[3]-poczatekCzesciCzyGrac[2])+"\n";    
    
    pString += "Runda 4: ";
    pString += (rozmiarGenomu-poczatekCzesciCzyGrac[3])+"\n";        
    
    return pString;
  }
  
  
  
}

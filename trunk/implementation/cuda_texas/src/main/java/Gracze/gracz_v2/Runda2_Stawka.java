package Gracze.gracz_v2;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import engine.Gra;
import engine.Karta;
import engine.RegulyGry;
import engine.rezultaty.Rezultat;

public class Runda2_Stawka {

  private Gra gra;
  private int przesuniecie;
  private int kolejnosc;
  private EvBinaryVectorIndividual individual;
  private Rezultat rezultat;
  
  

  /************** NUMERACJA GENOW **************/
  static final int JEST_PARA = 0;
  static final int JEST_PARA_SIZE = JEST_PARA + 1;
  static final int JEST_PARA_SIZE_LENGTH = 2;
  static final int JEST_TROJKA = JEST_PARA_SIZE + JEST_PARA_SIZE_LENGTH;
  static final int JEST_TROJKA_SIZE = JEST_PARA + 1;
  static final int JEST_TROJKA_SIZE_LENGTH = 2;  
  static final int SA_2PARY = JEST_TROJKA_SIZE + JEST_TROJKA_SIZE_LENGTH;
  static final int SA_2PARY_SIZE = SA_2PARY + 1;
  static final int SA_2PARY_SIZE_LENGTH = 2;    
  static final int JEST_KOLOR = SA_2PARY_SIZE + SA_2PARY_SIZE_LENGTH;
  static final int JEST_KOLOR_SIZE = JEST_KOLOR + 1;
  static final int JEST_KOLOR_SIZE_LENGTH = 3;     
  static final int JEST_STREET = JEST_KOLOR_SIZE + JEST_KOLOR_SIZE_LENGTH;
  static final int JEST_STREET_SIZE = JEST_STREET + 1;
  static final int JEST_STREET_SIZE_LENGTH = 3;    
  static final int MINIMAL_BID = JEST_STREET_SIZE_LENGTH + JEST_STREET_SIZE;    
  static final int MINIMAL_BID3 = MINIMAL_BID + 1;   
  static final int MINIMAL_BID5 = MINIMAL_BID3 + 1; 
  static final int MINIMAL_BID10 = MINIMAL_BID5 + 1;
  public static final int LENGTH = MINIMAL_BID10 + 1;
  
  
  
  public Runda2_Stawka(Gra gra_, int kolejnosc_, EvBinaryVectorIndividual individual_, int przesuniecie_) {
    gra = gra_;
    kolejnosc = kolejnosc_;
    individual = individual_;
    przesuniecie = przesuniecie_;
  }  
  
  public double decyzja() {
    
    rezultat = pobierzPrognoze5();
    double stawka = 0.0d;
    
    if (geneTrue(JEST_PARA) && regula_jest_para()) {
      stawka += gra.minimal_bid * BinToInt.decode(individual, przesuniecie + JEST_PARA_SIZE, JEST_PARA_SIZE_LENGTH);
    }
    if (geneTrue(SA_2PARY) && regula_sa_2pary()) {
      stawka += gra.minimal_bid * BinToInt.decode(individual, przesuniecie + SA_2PARY_SIZE, SA_2PARY_SIZE_LENGTH);
    }    
    if (geneTrue(JEST_TROJKA) && regula_jest_trojka()) {
      stawka += gra.minimal_bid * BinToInt.decode(individual, przesuniecie + JEST_TROJKA_SIZE, JEST_TROJKA_SIZE_LENGTH);
    }        
    if (geneTrue(JEST_KOLOR) && regula_jest_kolor()) {
      stawka += gra.minimal_bid * BinToInt.decode(individual, przesuniecie + JEST_KOLOR_SIZE, JEST_KOLOR_SIZE_LENGTH);
    }       
    if (geneTrue(JEST_STREET) && regula_jest_street()) {
      stawka += gra.minimal_bid * BinToInt.decode(individual, przesuniecie + JEST_STREET_SIZE, JEST_STREET_SIZE_LENGTH);
    }      
    if (geneTrue(MINIMAL_BID) && regula_minimal_bid())
      stawka += gra.minimal_bid;
    if (geneTrue(MINIMAL_BID3) && regula_minimal_bid3())
      stawka += 3 * gra.minimal_bid;    
    if (geneTrue(MINIMAL_BID5) && regula_minimal_bid5())
      stawka += 5 * gra.minimal_bid;       
    if (geneTrue(MINIMAL_BID10) && regula_minimal_bid10())
      stawka += 10 * gra.minimal_bid;     
    
    return stawka;
  }
 
  
  private boolean regula_jest_para() {
    if (rezultat.poziom>1)
      return true;

    return false;
  }
  
  private boolean regula_sa_2pary() {
    if (rezultat.poziom>2)
      return true;

    return false;
  }  
  
  private boolean regula_jest_trojka() {
    if (rezultat.poziom>3)
      return true;

    return false;
  }    
  
  private boolean regula_jest_kolor() {
    if (rezultat.poziom>5)
      return true;

    return false;
  }      
  
  private boolean regula_jest_street() {
    if (rezultat.poziom>4)
      return true;

    return false;
  }      
  
  private boolean regula_minimal_bid() {
    if (gra.stawka == gra.minimal_bid)
      return true;

    return false;
  }      

  private boolean regula_minimal_bid3() {
    if (gra.stawka <= 3 * gra.minimal_bid)
      return true;

    return false;
  }     
  
  private boolean regula_minimal_bid5() {
    if (gra.stawka <= 5 * gra.minimal_bid)
      return true;

    return false;
  } 
  
  private boolean regula_minimal_bid10() {
    if (gra.stawka <= 10 * gra.minimal_bid)
      return true;

    return false;
  }        
  
  
  
  /***** METODY POMOCNICZE ******/
  
  private boolean geneTrue(int gene) {
    if (individual.getGene(przesuniecie + gene) == 1)
      return true;
    return false;
  }    
  
  private Rezultat pobierzPrognoze5() {
    Karta[] karty = new Karta[7];
    karty[0] =  gra.getPublicCard(0);
    karty[1] =  gra.getPublicCard(1);
    karty[2] =  gra.getPublicCard(2);
    karty[3] =  gra.getPrivateCard(kolejnosc, 0);
    karty[4] =  gra.getPrivateCard(kolejnosc, 1); 
    karty[5] = new Karta(0,0);
    karty[6] = new Karta(0,0);
    
    return RegulyGry.najlepsza_karta(karty);
  }    
  
  public String toString() {
    String ret = new String();
    ret += "Stawka\n";
    ret += individual.getGene(JEST_PARA) + " jest para, stawka " +BinToInt.decode(individual, przesuniecie + JEST_PARA_SIZE, JEST_PARA_SIZE_LENGTH)+ "\n";
    ret += individual.getGene(SA_2PARY) + " sa 2 pary, stawka " +BinToInt.decode(individual, przesuniecie + SA_2PARY_SIZE, SA_2PARY_SIZE_LENGTH)+ "\n";
    ret += individual.getGene(JEST_TROJKA) + " jest trojka, stawka " +BinToInt.decode(individual, przesuniecie + JEST_TROJKA_SIZE, JEST_TROJKA_SIZE_LENGTH)+ "\n"; 
    ret += individual.getGene(JEST_KOLOR) + " jest kolor, stawka " +BinToInt.decode(individual, przesuniecie + JEST_KOLOR_SIZE, JEST_KOLOR_SIZE_LENGTH)+ "\n";  
    ret += individual.getGene(JEST_STREET) + " jest street, stawka " +BinToInt.decode(individual, przesuniecie + JEST_STREET_SIZE, JEST_STREET_SIZE_LENGTH)+ "\n"; 
    ret += individual.getGene(MINIMAL_BID) + " minimal bid \n";
    ret += individual.getGene(MINIMAL_BID3) + " 3 minimal bid \n";
    ret += individual.getGene(MINIMAL_BID5) + " 5 minimal bid \n";
    ret += individual.getGene(MINIMAL_BID10) + " 10 minimal bid \n";
 
    return ret;
  }  
  
}

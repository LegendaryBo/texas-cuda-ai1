package Gracze.gracz_v2;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import engine.Gra;

public class Runda1_Stawka {

  private Gra gra;
  private int przesuniecie;
  private int kolejnosc;
  private EvBinaryVectorIndividual individual;
  
  public static final int PARA_BID = 0;
  public static final int PARA_BID_SIZE = PARA_BID + 1;
  public static final int PARA_BID_SIZE_LENGTH = 8;
  public static final int WYSOKA_KARTA = PARA_BID_SIZE + PARA_BID_SIZE_LENGTH;
  public static final int WYSOKA_KARTA_SIZE = WYSOKA_KARTA + 1;
  public static final int WYSOKA_KARTA_SIZE_LENGTH = 6;
  public static final int KOLOR = WYSOKA_KARTA_SIZE + WYSOKA_KARTA_SIZE_LENGTH;
  public static final int KOLOR_SIZE = KOLOR + 1;
  public static final int KOLOR_SIZE_LENGTH = 3;
  public static final int STALA_STAWKA_MALA = KOLOR_SIZE + KOLOR_SIZE_LENGTH;
  public static final int STALA_STAWKA_SREDNIA = STALA_STAWKA_MALA + 1;
  public static final int STALA_STAWKA_DUZA = STALA_STAWKA_SREDNIA + 1;
  public static final int LENGTH = STALA_STAWKA_DUZA + 1;
  
  public Runda1_Stawka(Gra gra_, int kolejnosc_, EvBinaryVectorIndividual individual_, int przesuniecie_) {
    gra = gra_;
    kolejnosc = kolejnosc_;
    individual = individual_;
    przesuniecie = przesuniecie_;
  }
  
  
  public double decyzja() {
    
    double stawka = 0;
    
    if (geneTrue(PARA_BID) && para_bid())
      stawka += gra.minimal_bid * BinToInt.decode(individual, przesuniecie + PARA_BID_SIZE, PARA_BID_SIZE_LENGTH);    
    if (geneTrue(WYSOKA_KARTA) && wysoka_karta_bid())
      stawka += gra.minimal_bid * BinToInt.decode(individual, przesuniecie + WYSOKA_KARTA_SIZE, WYSOKA_KARTA_SIZE_LENGTH);     
    if (geneTrue(KOLOR) && kolor_bid())
      stawka += gra.minimal_bid * BinToInt.decode(individual, przesuniecie + KOLOR_SIZE, KOLOR_SIZE_LENGTH);   
    if (geneTrue(STALA_STAWKA_MALA))
      stawka += gra.minimal_bid;
    if (geneTrue(STALA_STAWKA_SREDNIA))
      stawka += 3 * gra.minimal_bid;    
    if (geneTrue(STALA_STAWKA_DUZA))
      stawka += 10 * gra.minimal_bid;        
    
    return stawka;
  }
  
  /****** REGULY ******/
  private boolean para_bid() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra.getPrivateCard(kolejnosc, 1).wysokosc)
      return true;
    else 
      return false;    
  }  

  private boolean wysoka_karta_bid() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc > 11 &&
        gra.getPrivateCard(kolejnosc, 1).wysokosc > 11)
      return true;
    else 
      return false;    
  }    
  
  private boolean kolor_bid() {
    if (gra.getPrivateCard(kolejnosc, 0).kolor ==
        gra.getPrivateCard(kolejnosc, 1).kolor)
      return true;
    else 
      return false;    
  }    
  
  
  /***** METODY POMOCNICZE ******/
  
  private boolean geneTrue(int gene) {
    if (individual.getGene(przesuniecie + gene) == 1)
      return true;
    return false;
  }  
  
  public String toString() {
    
    String ret = new String();
    ret += "STAWKA\n";
    ret += individual.getGene(przesuniecie + PARA_BID) + " para stawka: " + BinToInt.decode(individual, przesuniecie + PARA_BID_SIZE, PARA_BID_SIZE_LENGTH) + " * minimum \n";
    ret += individual.getGene(przesuniecie + WYSOKA_KARTA) + " wysoka karta stawka: " + BinToInt.decode(individual, przesuniecie + WYSOKA_KARTA_SIZE, WYSOKA_KARTA_SIZE_LENGTH) + " * minimum \n";
    ret += individual.getGene(przesuniecie + KOLOR) + " kolor stawka: " + BinToInt.decode(individual, przesuniecie + KOLOR_SIZE, KOLOR_SIZE_LENGTH) + " * minimum \n";
    ret += individual.getGene(przesuniecie + STALA_STAWKA_MALA) + " stawka minimum\n";
    ret += individual.getGene(przesuniecie + STALA_STAWKA_SREDNIA) + " stawka minimum * 3\n";
    ret += individual.getGene(przesuniecie + STALA_STAWKA_DUZA) + " stawka minimum * 10\n";
 
    
    return ret;    
    
  }
  
  
}

package Gracze.gracz_v2;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import engine.Gra;
import engine.rezultaty.Rezultat;

public class Runda1_CzyGrac {

  
  private Gra gra = null;
  private int kolejnosc;
  private EvBinaryVectorIndividual individual = null;
  private int przesuniecie;
  
  /************** NUMERACJA GENOW **************/
  static final int ZAWSZE_GRAC = 0;
  static final int KARTY_TEJ_SAMEJ_WAGI = ZAWSZE_GRAC + 1;
  static final int KARTY_TEGO_SAMEGO_KOLORU = KARTY_TEJ_SAMEJ_WAGI + 1;
  static final int STAWKA_MINIMUM = KARTY_TEGO_SAMEGO_KOLORU + 1;
  static final int STAWKA_XXX = STAWKA_MINIMUM + 1;
  static final int STAWKA_XXX_WYSOKOSC_STAWKI = STAWKA_XXX + 1;
  static final int STAWKA_XXX_WYSOKOSC_STAWKI_LENGTH = 6;
  static final int WYSOKIE_KARTY = STAWKA_XXX_WYSOKOSC_STAWKI + STAWKA_XXX_WYSOKOSC_STAWKI_LENGTH;
  static final int BARDZO_WYSOKIE_KARTY = WYSOKIE_KARTY + 1;  
  static final int WYMAGANYCH_GLOSOW = BARDZO_WYSOKIE_KARTY + 1;
  static final int WYMAGANYCH_GLOSOW_LENGTH = 3;
  public static final int LENGTH = WYMAGANYCH_GLOSOW + WYMAGANYCH_GLOSOW_LENGTH + 1;
  
  public Runda1_CzyGrac(Gra gra_, int kolejnosc_, EvBinaryVectorIndividual individual_, int przesuniecie_) {
    gra = gra_;
    kolejnosc = kolejnosc_;
    individual = individual_;
    przesuniecie = przesuniecie_;
  }
  
  
  public boolean decyzja(RegulaAbstrakcyjna[] reguly, Rezultat rezultat) {
    
    int glosow = 0;
    
    for (int i=0; i < reguly.length; i++) {
        glosow += reguly[i].aplikujRegule(gra, kolejnosc, individual,rezultat);
    }
     
//    if (geneTrue(KARTY_TEGO_SAMEGO_KOLORU) && regula_2karty_tego_samego_koloru())
//      glosow++;       
//    if (geneTrue(STAWKA_MINIMUM) && regula_tylko_stawka_minimum())
//      glosow++;       
//    if (geneTrue(STAWKA_XXX) && regula_tylko_stawka_XXX(STAWKA_XXX_WYSOKOSC_STAWKI))
//      glosow++;           
//    if (geneTrue(WYSOKIE_KARTY) && regula_wysokie_karty())
//      glosow++;       
//    if (geneTrue(BARDZO_WYSOKIE_KARTY) && regula_bardzo_wysokie_karty())
//      glosow++;       
    
    // ostateczna decyzja
    if (wymaganych_glosow(WYMAGANYCH_GLOSOW, WYMAGANYCH_GLOSOW_LENGTH) <= glosow)
      return true;
    else return false;    
    
  }
  
  
  /***** REGULY *****/
  
  
  private boolean regula_2karty_tej_samej_wagi() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra.getPrivateCard(kolejnosc, 1).wysokosc)
      return true;
    else 
      return false;
  }  
  
  private boolean regula_2karty_tego_samego_koloru() {
    if (gra.getPrivateCard(kolejnosc, 0).kolor == gra.getPrivateCard(kolejnosc, 1).kolor)
      return true;
    else 
      return false;
  }  
  
  private boolean regula_tylko_stawka_minimum() {
    if (gra.stawka == gra.minimal_bid)
      return true;
    else 
      return false;
  }      
 
  private boolean regula_tylko_stawka_XXX(int gen) {  
  if (gra.stawka < BinToInt.decode(individual, przesuniecie + gen, STAWKA_XXX_WYSOKOSC_STAWKI_LENGTH) * gra.minimal_bid)
    return true;
  else 
    return false;
  }       
  
  // kazda karta powyzej 10 
  private boolean regula_wysokie_karty() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc > 10 && 
        gra.getPrivateCard(kolejnosc, 1).wysokosc > 10)
      return true;
    else 
      return false;
  }    
  
  private boolean regula_bardzo_wysokie_karty() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc > 12 && 
        gra.getPrivateCard(kolejnosc, 1).wysokosc > 12)
      return true;
    else 
      return false;
  }    
  
  
  
  
  private int wymaganych_glosow(int pozycja, int length) {
    return BinToInt.decode(individual, przesuniecie + pozycja, length);
  }
  
  /***** METODY POMOCNICZE ******/
  
  private boolean geneTrue(int gene) {
    if (individual.getGene(przesuniecie + gene) == 1)
      return true;
    return false;
  }
  
  
  
  /*** specyfikacja gry ***/
  public String toString() {
    
    String ret = new String();
    ret += "RUNDA1:\n";
    ret += "REGULY\n";
    ret += individual.getGene(przesuniecie + ZAWSZE_GRAC) + " zawsze grac\n";
    ret += individual.getGene(przesuniecie + KARTY_TEJ_SAMEJ_WAGI) + " 2 karty tej samej wagi\n";
    ret += individual.getGene(przesuniecie + KARTY_TEGO_SAMEGO_KOLORU) + " 2 karty tego samego koloru\n";
    ret += individual.getGene(przesuniecie + STAWKA_MINIMUM) + " tylko stawka minimum\n";
    ret += individual.getGene(przesuniecie + STAWKA_XXX) + " stawka co najwyzej " + BinToInt.decode(individual, przesuniecie + STAWKA_XXX_WYSOKOSC_STAWKI, STAWKA_XXX_WYSOKOSC_STAWKI_LENGTH) + " minimum \n";
    ret += individual.getGene(przesuniecie + WYSOKIE_KARTY) + " karty powyzej dziesiatek\n";
    ret += individual.getGene(przesuniecie + BARDZO_WYSOKIE_KARTY) + " karty powyzej dam\n";
    ret += "Wymaganych glosow: "+BinToInt.decode(individual, przesuniecie + WYMAGANYCH_GLOSOW, WYMAGANYCH_GLOSOW_LENGTH)+" \n";
 
    
    return ret;
    
  }
  
  
}

package Gracze.gracz_v2;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import engine.Gra;

public class Runda1_Dobijanie {

  private Gra gra;
  private int przesuniecie;
  private int kolejnosc;
  private EvBinaryVectorIndividual individual;    
  
  public static final int DOBIJAC_ZAWSZE = 0;
  public static final int DOBIJAC_POLOWA = DOBIJAC_ZAWSZE + 1;
  public static final int DOBIJAC_CWIARA = DOBIJAC_POLOWA + 1;  
  public static final int DOBIJAC_DYCHA = DOBIJAC_CWIARA + 1; 
  public static final int DOBIJAC_GDY_PARA_ZAWSZE = DOBIJAC_DYCHA + 1; 
  public static final int DOBIJAC_GDY_PARA_POLOWA = DOBIJAC_GDY_PARA_ZAWSZE + 1; 
  public static final int DOBIJAC_GDY_PARA_DYCHA = DOBIJAC_GDY_PARA_POLOWA + 1; 
  public static final int DOBIJAC_GDY_WYSOKA = DOBIJAC_GDY_PARA_DYCHA + 1;
  public static final int LENGTH = DOBIJAC_GDY_WYSOKA + 1; 
  
  public Runda1_Dobijanie(Gra gra_, int kolejnosc_, EvBinaryVectorIndividual individual_, int przesuniecie_) {
    gra = gra_;
    kolejnosc = kolejnosc_;
    individual = individual_;
    przesuniecie = przesuniecie_;
  }  
  
  public boolean decyzja(double stawka) {
    
    if (geneTrue(DOBIJAC_ZAWSZE))
      return true;
    if (geneTrue(DOBIJAC_POLOWA) && stawka < gra.stawka * 0.5)
      return true;    
    if (geneTrue(DOBIJAC_CWIARA) && stawka < gra.stawka * 0.25)
      return true;        
    if (geneTrue(DOBIJAC_DYCHA) && stawka < gra.stawka * 0.1)
      return true;      
    if (geneTrue(DOBIJAC_GDY_PARA_ZAWSZE) && para_bid())
      return true;      
    if (geneTrue(DOBIJAC_GDY_PARA_POLOWA) && para_bid() && stawka < gra.stawka * 0.5)
      return true;       
    if (geneTrue(DOBIJAC_GDY_PARA_DYCHA) && para_bid() && stawka < gra.stawka * 0.1)
      return true;         
    if (geneTrue(DOBIJAC_GDY_WYSOKA) && wysoka() && stawka < gra.stawka * 0.33)
      return true;           
    
    
    return false;
    
  }
  
  
  
  
  private boolean para_bid() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra.getPrivateCard(kolejnosc, 1).wysokosc)
      return true;
    else 
      return false;    
  }  
  
  private boolean wysoka() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc > 11 &&
        gra.getPrivateCard(kolejnosc, 1).wysokosc > 11)
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
    ret += "DOBIJANIE\n";
    ret += individual.getGene(przesuniecie + DOBIJAC_ZAWSZE) + " dobijac zawsze \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_POLOWA) + " dobijac gdy przebito dwukrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_CWIARA) + " dobijac gdy przebito czterokrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_DYCHA) + " dobijac gdy przebito dziesieciokrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_PARA_ZAWSZE) + " dobijac gdy para \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_PARA_POLOWA) + " dobijac gdy jest para i przebito dwukrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_PARA_DYCHA) + " dobijac gdy jest para i przebito dziesieciokrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_WYSOKA) + " dobijac gdy jest wysoka karta i przebito sdwukrotnie \n";
    
    return ret;    
    
  }  
  
  
  
  
  
}

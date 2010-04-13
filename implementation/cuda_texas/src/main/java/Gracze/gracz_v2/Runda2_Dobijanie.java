package Gracze.gracz_v2;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import engine.Gra;
import engine.Karta;
import engine.RegulyGry;
import engine.rezultaty.Rezultat;

public class Runda2_Dobijanie {

  private Gra gra;
  private int przesuniecie;
  private int kolejnosc;
  private EvBinaryVectorIndividual individual;    
  private Rezultat rezultat = null;
  
  public static final int DOBIJAC_ZAWSZE = 0;
  public static final int DOBIJAC_POLOWA = DOBIJAC_ZAWSZE + 1;
  public static final int DOBIJAC_CWIARA = DOBIJAC_POLOWA + 1;  
  public static final int DOBIJAC_DYCHA = DOBIJAC_CWIARA + 1; 
  public static final int DOBIJAC_GDY_PARA_ZAWSZE = DOBIJAC_DYCHA + 1; 
  public static final int DOBIJAC_GDY_PARA_POLOWA = DOBIJAC_GDY_PARA_ZAWSZE + 1; 
  public static final int DOBIJAC_GDY_PARA_DYCHA = DOBIJAC_GDY_PARA_POLOWA + 1; 
  
  public static final int DOBIJAC_GDY_2PARY_ZAWSZE = DOBIJAC_GDY_PARA_DYCHA + 1; 
  public static final int DOBIJAC_GDY_2PARY_POLOWA = DOBIJAC_GDY_2PARY_ZAWSZE + 1; 
  public static final int DOBIJAC_GDY_2PARY_DYCHA = DOBIJAC_GDY_2PARY_POLOWA + 1;   
  
  public static final int DOBIJAC_GDY_TROJKA_ZAWSZE = DOBIJAC_GDY_2PARY_DYCHA + 1; 
  public static final int DOBIJAC_GDY_TROJKA_POLOWA = DOBIJAC_GDY_TROJKA_ZAWSZE + 1; 
  public static final int DOBIJAC_GDY_TROJKA_DYCHA = DOBIJAC_GDY_TROJKA_POLOWA + 1;    
  
  public static final int DOBIJAC_GDY_FULL_ZAWSZE = DOBIJAC_GDY_TROJKA_DYCHA + 1; 
  public static final int DOBIJAC_GDY_FULL_POLOWA = DOBIJAC_GDY_FULL_ZAWSZE + 1; 
  public static final int DOBIJAC_GDY_FULL_DYCHA = DOBIJAC_GDY_FULL_POLOWA + 1;     
  
  public static final int DOBIJAC_GDY_WYSOKA = DOBIJAC_GDY_FULL_DYCHA + 1;
  public static final int LENGTH = DOBIJAC_GDY_WYSOKA + 1; 
  
  public Runda2_Dobijanie(Gra gra_, int kolejnosc_, EvBinaryVectorIndividual individual_, int przesuniecie_) {
    gra = gra_;
    kolejnosc = kolejnosc_;
    individual = individual_;
    przesuniecie = przesuniecie_;
  }  
  
  public boolean decyzja(double stawka) {
    
    rezultat = pobierzPrognoze5();
    
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
    
    if (geneTrue(DOBIJAC_GDY_2PARY_ZAWSZE) && pary2_bid())
      return true;      
    if (geneTrue(DOBIJAC_GDY_2PARY_POLOWA) && pary2_bid() && stawka < gra.stawka * 0.5)
      return true;       
    if (geneTrue(DOBIJAC_GDY_2PARY_DYCHA) && pary2_bid() && stawka < gra.stawka * 0.1)
      return true;          
    
    if (geneTrue(DOBIJAC_GDY_TROJKA_ZAWSZE) && trojka_bid())
      return true;      
    if (geneTrue(DOBIJAC_GDY_TROJKA_POLOWA) && trojka_bid() && stawka < gra.stawka * 0.5)
      return true;       
    if (geneTrue(DOBIJAC_GDY_TROJKA_DYCHA) && trojka_bid() && stawka < gra.stawka * 0.1)
      return true;          
    
    if (geneTrue(DOBIJAC_GDY_FULL_ZAWSZE) && co_najmniej_street_bid())
      return true;      
    if (geneTrue(DOBIJAC_GDY_FULL_POLOWA) && co_najmniej_street_bid() && stawka < gra.stawka * 0.5)
      return true;       
    if (geneTrue(DOBIJAC_GDY_FULL_DYCHA) && co_najmniej_street_bid() && stawka < gra.stawka * 0.1)
      return true;         
    
    
    if (geneTrue(DOBIJAC_GDY_WYSOKA) && wysoka() && stawka < gra.stawka * 0.33)
      return true;           
    
    
    return false;
    
  }
  
  
  
  
  private boolean para_bid() {
    if (rezultat.poziom == 2)
      return true;
    else 
      return false;    
  }  
  
  private boolean pary2_bid() {
    if (rezultat.poziom == 3)
      return true;
    else 
      return false;    
  }   
  
  private boolean trojka_bid() {
    if (rezultat.poziom == 4)
      return true;
    else 
      return false;    
  }    
  
  private boolean co_najmniej_street_bid() {
    if (rezultat.poziom >= 5)
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
    ret += "DOBIJANIE\n";
    ret += individual.getGene(przesuniecie + DOBIJAC_ZAWSZE) + " dobijac zawsze \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_POLOWA) + " dobijac gdy przebito dwukrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_CWIARA) + " dobijac gdy przebito czterokrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_DYCHA) + " dobijac gdy przebito dziesieciokrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_PARA_ZAWSZE) + " dobijac gdy para \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_PARA_POLOWA) + " dobijac gdy jest para i przebito dwukrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_PARA_DYCHA) + " dobijac gdy jest para i przebito dziesieciokrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_2PARY_ZAWSZE) + " dobijac gdy 2 pary  \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_2PARY_POLOWA) + " dobijac gdy 2 pary i przebito dwukrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_2PARY_DYCHA) + " dobijac gdy 2 pary i przebito dziesieciokrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_TROJKA_ZAWSZE) + " dobijac gdy trojka  \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_TROJKA_POLOWA) + " dobijac gdy trojka i przebito dwukrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_TROJKA_DYCHA) + " dobijac gdy trojka i przebito dziesieciokrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_FULL_ZAWSZE) + " dobijac gdy co najmniej street  \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_FULL_POLOWA) + " dobijac gdy co najmniej street i przebito dwukrotnie \n";
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_FULL_DYCHA) + " dobijac gdy co najmniej street i przebito dziesieciokrotnie \n";    
    ret += individual.getGene(przesuniecie + DOBIJAC_GDY_WYSOKA) + " dobijac gdy jest wysoka karta i przebito sdwukrotnie \n";
    
    return ret;    
    
  }  
  
  
  
  
  
}

package wevo;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

/**
 * 
 * @author Kacper Gorski 
 *
 *@deprecated wersja 0.1, brzydka i smiedzi
 */
public class TexasIndividual {

  public static String describe(EvBinaryVectorIndividual individual) {
    
    String ret = new String("Runda1: ");
    
    int regul_do_wejscia_runda1 = 0;
    int wymaganych_regul1 = 0;
    int regul_stawek_runda1 = 0;
    int regul_dobijania_runda1 = 0;

    int regul_do_wejscia_runda2 = 0;
    int wymaganych_regul2 = 0;
    int regul_stawek_runda2 = 0;
    int regul_dobijania_runda2 = 0;    
    int regul_pass_runda2 = 0;
    
    int regul_do_wejscia_runda3 = 0;
    int wymaganych_regul3 = 0;
    int regul_stawek_runda3 = 0;
    int regul_dobijania_runda3 = 0;    
    int regul_pass_runda3 = 0;
    
    int regul_do_wejscia_runda4 = 0;
    int wymaganych_regul4 = 0;
    int regul_stawek_runda4 = 0;
    int regul_dobijania_runda4 = 0;    
    int regul_pass_runda4 = 0;    
    
    for (int i=0; i < 7; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_do_wejscia_runda1++;
    }
    if (individual.getGene(7) == 1 && wymaganych_regul1==0)
      wymaganych_regul1 = 1;
    if (individual.getGene(8) == 1 && wymaganych_regul1==0)
      wymaganych_regul1 = 3;
    if (individual.getGene(9) == 1 && wymaganych_regul1==0)
      wymaganych_regul1 = 5;    
    
    ret += " ";
    for (int i=7; i < 10; i++) {
      ret += individual.getGene(i);
    }
    ret += " ";
    for (int i=10; i < 18; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_stawek_runda1++;
    }
    ret += " ";
    for (int i=18; i < 21; i++) {
      ret += individual.getGene(i);   
      if (individual.getGene(i) == 1)
        regul_dobijania_runda1++;
    }
    ret += "\n";
    
    ret += "rozpatrywanych regul do wejscia - "+regul_do_wejscia_runda1+" niezbedna ilosc regul do wejscia - "+wymaganych_regul1;
    ret += " regu³ do stawek - "+regul_stawek_runda1+" Regul dobijania"+regul_dobijania_runda1;    
    ret += "\n";
    ret +="Runda2: ";
    
    for (int i=21; i < 33; i++) {
      ret += individual.getGene(i);  
      if (individual.getGene(i) == 1)
        regul_do_wejscia_runda2++;      
    }
    ret += " ";
    for (int i=33; i < 36; i++) {
      ret += individual.getGene(i);
    }
    
    if (individual.getGene(33) == 1  && wymaganych_regul2==0)
      wymaganych_regul2 = 1;
    if (individual.getGene(34) == 3  && wymaganych_regul2==0)
      wymaganych_regul2 = 3;
    if (individual.getGene(35) == 6  && wymaganych_regul2==0)
      wymaganych_regul2 = 8;        
    
    ret += " ";
    for (int i=36; i < 43; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_stawek_runda2++;
    }
    ret += " ";
    for (int i=43; i < 47; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_dobijania_runda2++;
    }
    ret += " ";
    for (int i=47; i < 49; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_pass_runda2++;
    }    
   
    ret += "\n";
    ret += "rozpatrywanych regul do wejscia - "+regul_do_wejscia_runda2+" niezbedna ilosc regul do wejscia - "+wymaganych_regul2;
    ret += " regu³ do stawek - "+regul_stawek_runda2+" Regul dobijania"+regul_dobijania_runda2+" Regul pasowania"+regul_pass_runda2;           
    ret += "\n";
    ret +="Runda3: ";


    for (int i=49; i < 61; i++) {
      ret += individual.getGene(i);  
      if (individual.getGene(i) == 1)
        regul_do_wejscia_runda3++;      
    }
    ret += " ";
    for (int i=63; i < 66; i++) {
      ret += individual.getGene(i);
    }
    
    if (individual.getGene(63) == 1  && wymaganych_regul3==0)
      wymaganych_regul3 = 1;
    if (individual.getGene(64) == 3  && wymaganych_regul3==0)
      wymaganych_regul3 = 3;
    if (individual.getGene(65) == 6  && wymaganych_regul3==0)
      wymaganych_regul3 = 8;        
    
    ret += " ";
    for (int i=66; i < 73; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_stawek_runda3++;
    }
    ret += " ";
    for (int i=73; i < 77; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_dobijania_runda3++;
    }
    ret += " ";
    for (int i=61; i < 63; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_pass_runda3++;
    }        
   
    ret += "\n";
    ret += "rozpatrywanych regul do wejscia - "+regul_do_wejscia_runda3+" niezbedna ilosc regul do wejscia - "+wymaganych_regul3;
    ret += " regu³ do stawek - "+regul_stawek_runda3+" Regul dobijania"+regul_dobijania_runda3 + " Regul pasowania + "+regul_pass_runda3;    

    ret += "\n";
    ret +="Runda4: ";


    for (int i=77; i < 89; i++) {
      ret += individual.getGene(i);  
      if (individual.getGene(i) == 1)
        regul_do_wejscia_runda4++;      
    }
    ret += " ";
    for (int i=91; i < 94; i++) {
      ret += individual.getGene(i);
    }
    
    if (individual.getGene(91) == 1  && wymaganych_regul4==0)
      wymaganych_regul4 = 1;
    if (individual.getGene(92) == 3  && wymaganych_regul4==0)
      wymaganych_regul4 = 3;
    if (individual.getGene(93) == 6  && wymaganych_regul4==0)
      wymaganych_regul4 = 8;        
    
    ret += " ";
    for (int i=94; i < 101; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_stawek_runda4++;
    }
    ret += " ";
    for (int i=101; i < 105; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_dobijania_runda4++;
    }
    ret += " ";
    for (int i=89; i < 91; i++) {
      ret += individual.getGene(i);
      if (individual.getGene(i) == 1)
        regul_pass_runda4++;
    }        
   
    ret += "\n";
    ret += "rozpatrywanych regul do wejscia - "+regul_do_wejscia_runda4+" niezbedna ilosc regul do wejscia - "+wymaganych_regul4;
    ret += " regu³ do stawek - "+regul_stawek_runda4+" Regul dobijania"+regul_dobijania_runda4 + " Regul pasowania + "+regul_pass_runda4;    
    
    
    return ret;
  }
  
}

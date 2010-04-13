package Gracze;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import engine.Karta;
import engine.RegulyGry;
import engine.rezultaty.Rezultat;

public class GraczAI extends Gracz {

  private EvBinaryVectorIndividual individual = null;
  private int kolejnosc=0;
  public boolean pass = false;
  private Rezultat rezultat = null;
  
  public GraczAI(EvBinaryVectorIndividual individual_, int kolejnosc_) {
    individual = individual_;
    kolejnosc = kolejnosc_;
  }
  
  @Override
  public double play(int i, double bid) {
    
    if (i == 0 && !czy_grac_runda1()) {
      pass = true;
      return -1;

    }
    if (i == 0 && czy_grac_runda1()) {
      //double stawka = stawka_runda1();
      double stawka = stawka_runda1();
      
      if (stawka < gra.stawka) {
        if (dobic_do_stawki(stawka))
          stawka = gra.stawka;
        else {
          pass = true;
          return -1;
        }
      }
      
      bilans -= stawka - bid;  
      if (musik > 0) {
        bilans +=musik;
        musik = 0;
      }
      return stawka;
    }
    
    if (i==1)
      rezultat = pobierzPrognoze5();
    
    if (i == 1 && !czy_grac_runda2()) {
      pass = true;
      return -1;

    }
    if (i == 1 && czy_grac_runda2()) {
      //double stawka = stawka_runda1();
      double stawka = stawka_runda2();
      
      if (stawka < gra.stawka) {
        if (dobic_do_stawki_runda2(stawka))
          stawka = gra.stawka;
        else {
          pass = true;
          return -1;
        }
      }
      
      bilans -= stawka - bid;  
      if (musik > 0) {
        bilans +=musik;
        musik = 0;
      }
      return stawka;
    }    
    
    if (i==2)
      rezultat = pobierzPrognoze6();
    
    if (i == 2 && !czy_grac_runda3()) {
      pass = true;
      return -1;

    }
    if (i == 2 && czy_grac_runda3()) {
      //double stawka = stawka_runda1();
      double stawka = stawka_runda3();
      
      if (stawka < gra.stawka) {
        if (dobic_do_stawki_runda3(stawka))
          stawka = gra.stawka;
        else {
          pass = true;
          return -1;
        }
      }
      
      bilans -= stawka - bid;  
      if (musik > 0) {
        bilans +=musik;
        musik = 0;
      }
      return stawka;
    }        
    
    if (i==3)
      rezultat = pobierzPrognoze7();
    
    if (i == 3 && !czy_grac_runda4()) {
      pass = true;
      return -1;

    }
    if (i == 3 && czy_grac_runda4()) {
      //double stawka = stawka_runda1();
      double stawka = stawka_runda4();
      
      if (stawka < gra.stawka) {
        if (dobic_do_stawki_runda4(stawka))
          stawka = gra.stawka;
        else {
          pass = true;
          return -1;
        }
      }
      
      bilans -= stawka - bid;  
      if (musik > 0) {
        bilans +=musik;
        musik = 0;
      }
      return stawka;
    }            
    
    throw new IllegalStateException("tego byc nie powinno...");

    
    
  }

  
  private boolean czy_grac_runda4() {
    int glosow = 0;
    
    if (individual.getGene(77) == 1 && czy_grac_2runda_zawsze_grac())
      glosow++;        
    if (individual.getGene(78) == 1 && czy_grac_2runda_jest_para())
      glosow++;         
    if (individual.getGene(79) == 1 && czy_grac_2runda_jest_2pary())
      glosow++;         
    if (individual.getGene(80) == 1 && czy_grac_2runda_jest_trojka())
      glosow++;            
    if (individual.getGene(81) == 1 && czy_grac_2runda_jest_street())
      glosow++;          
    if (individual.getGene(82) == 1 && czy_grac_2runda_jest_4gracz())
      glosow++;           
    if (individual.getGene(83) == 1 && czy_grac_2runda_jest_conajwyzej_3graczy())
      glosow++;        
    if (individual.getGene(84) == 1 && czy_grac_runda2_tylko_stawka_minimum())
      glosow++;        
    if (individual.getGene(85) == 1 && czy_grac_runda2_co_najwyzej_2minimum())
      glosow++;       
    if (individual.getGene(86) == 1 && czy_grac_runda2_co_najwyzej_5minimum())
      glosow++;       
    if (individual.getGene(87) == 1 && czy_grac_runda2_co_najwyzej_10minimum())
      glosow++;     
    if (individual.getGene(88) == 1 && czy_grac_runda2_co_najmniej10())
      glosow++;         
    
    if (individual.getGene(89) == 1 && czy_pass_runda2_tak_gdy_jest_smiec())
      return true;    
    if (individual.getGene(90) == 1 && czy_pass_runda2_tak_gdy_jest_para())
      return true;      
    
    if (individual.getGene(91) == 1 && czy_grac_runda2_tak_gdy_jest_glos(glosow))
      return true;
    if (individual.getGene(92) == 1 && czy_grac_runda2_tak_gdy_sa_3glosy(glosow))
      return true;
    if (individual.getGene(93) == 1 && czy_grac_runda2_tak_gdy_sa_7glosy(glosow))
      return true;    
    
    return false;
  }    
  
  
  private boolean czy_grac_runda3() {
    int glosow = 0;
    
    if (individual.getGene(49) == 1 && czy_grac_2runda_zawsze_grac())
      glosow++;        
    if (individual.getGene(50) == 1 && czy_grac_2runda_jest_para())
      glosow++;         
    if (individual.getGene(51) == 1 && czy_grac_2runda_jest_2pary())
      glosow++;         
    if (individual.getGene(52) == 1 && czy_grac_2runda_jest_trojka())
      glosow++;            
    if (individual.getGene(53) == 1 && czy_grac_2runda_jest_street())
      glosow++;          
    if (individual.getGene(54) == 1 && czy_grac_2runda_jest_4gracz())
      glosow++;           
    if (individual.getGene(55) == 1 && czy_grac_2runda_jest_conajwyzej_3graczy())
      glosow++;        
    if (individual.getGene(56) == 1 && czy_grac_runda2_tylko_stawka_minimum())
      glosow++;        
    if (individual.getGene(57) == 1 && czy_grac_runda2_co_najwyzej_2minimum())
      glosow++;       
    if (individual.getGene(58) == 1 && czy_grac_runda2_co_najwyzej_5minimum())
      glosow++;       
    if (individual.getGene(59) == 1 && czy_grac_runda2_co_najwyzej_10minimum())
      glosow++;     
    if (individual.getGene(60) == 1 && czy_grac_runda2_co_najmniej10())
      glosow++;         
    
    if (individual.getGene(61) == 1 && czy_pass_runda2_tak_gdy_jest_smiec())
      return true;    
    if (individual.getGene(62) == 1 && czy_pass_runda2_tak_gdy_jest_para())
      return true;      
    
    if (individual.getGene(63) == 1 && czy_grac_runda2_tak_gdy_jest_glos(glosow))
      return true;
    if (individual.getGene(64) == 1 && czy_grac_runda2_tak_gdy_sa_3glosy(glosow))
      return true;
    if (individual.getGene(65) == 1 && czy_grac_runda2_tak_gdy_sa_7glosy(glosow))
      return true;    
    
    return false;
  }  
  
  
  private Rezultat pobierzPrognoze6() {
    Karta[] karty = new Karta[7];
    karty[0] =  gra.getPublicCard(0);
    karty[1] =  gra.getPublicCard(1);
    karty[2] =  gra.getPublicCard(2);
    karty[3] =  gra.getPrivateCard(kolejnosc, 0);
    karty[4] =  gra.getPrivateCard(kolejnosc, 1); 
    karty[5] =  gra.getPublicCard(3);
    karty[6] = new Karta(0,0);
    
    return RegulyGry.najlepsza_karta(karty);
  }
  
  private Rezultat pobierzPrognoze7() {
    Karta[] karty = new Karta[7];
    karty[0] =  gra.getPublicCard(0);
    karty[1] =  gra.getPublicCard(1);
    karty[2] =  gra.getPublicCard(2);
    karty[3] =  gra.getPrivateCard(kolejnosc, 0);
    karty[4] =  gra.getPrivateCard(kolejnosc, 1); 
    karty[5] =  gra.getPublicCard(3);
    karty[6] = gra.getPublicCard(4);
    
    return RegulyGry.najlepsza_karta(karty);
  }

  private boolean dobic_do_stawki_runda3(double propozycja) {
    if (individual.getGene(73) == 1 && czy_dobic_runda2_zawsze())
      return true;
    if (individual.getGene(74) == 1 && czy_dobic_runda2_gry_brakuje_cwiary(propozycja, gra.stawka))
      return true;
    if (individual.getGene(75) == 1 && czy_dobic_runda2_gry_brakuje_polowy(propozycja, gra.stawka))
      return true;    
    if (individual.getGene(76) == 1 && czy_dobic_runda2_gry_brakuje_3_4(propozycja, gra.stawka))
      return true;        
    return false;    
  }  
  
  private boolean dobic_do_stawki_runda4(double propozycja) {
    if (individual.getGene(101) == 1 && czy_dobic_runda2_zawsze())
      return true;
    if (individual.getGene(102) == 1 && czy_dobic_runda2_gry_brakuje_cwiary(propozycja, gra.stawka))
      return true;
    if (individual.getGene(103) == 1 && czy_dobic_runda2_gry_brakuje_polowy(propozycja, gra.stawka))
      return true;    
    if (individual.getGene(104) == 1 && czy_dobic_runda2_gry_brakuje_3_4(propozycja, gra.stawka))
      return true;        
    return false;    
  }    
  
  
  private boolean dobic_do_stawki_runda2(double propozycja) {
    if (individual.getGene(43) == 1 && czy_dobic_runda2_zawsze())
      return true;
    if (individual.getGene(44) == 1 && czy_dobic_runda2_gry_brakuje_cwiary(propozycja, gra.stawka))
      return true;
    if (individual.getGene(45) == 1 && czy_dobic_runda2_gry_brakuje_polowy(propozycja, gra.stawka))
      return true;    
    if (individual.getGene(46) == 1 && czy_dobic_runda2_gry_brakuje_3_4(propozycja, gra.stawka))
      return true;        
    return false;    
  }

  public boolean czy_dobic_runda2_zawsze() {
    return true;
  }
  
  public boolean czy_dobic_runda2_gry_brakuje_cwiary(double propozycja, double stawka) {
    if (propozycja / stawka > 0.75)
      return true;
    else
      return false;
  } 
  
  public boolean czy_dobic_runda2_gry_brakuje_polowy(double propozycja, double stawka) {
    if (propozycja / stawka > 0.5)
      return true;
    else
      return false;
  }     
  
  public boolean czy_dobic_runda2_gry_brakuje_3_4(double propozycja, double stawka) {
    if (propozycja / stawka > 0.25)
      return true;
    else
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
  
  private double stawka_runda2_para_bid_minimum() {
    if (rezultat.poziom>0)
      return gra.minimal_bid;
    else 
      return 0.0;    
  }

  private double stawka_runda2_para_bid_5minimum() {
    if (rezultat.poziom>0)
      return 5*gra.minimal_bid;
    else 
      return 0.0;    
  }
 
  private double stawka_runda2_2pary_bid_minimum() {
    if (rezultat.poziom>1)
      return gra.minimal_bid;
    else 
      return 0.0;    
  }

  private double stawka_runda2_2pary_bid_5minimum() {
    if (rezultat.poziom>1)
      return 5*gra.minimal_bid;
    else 
      return 0.0;    
  }  
  
  private double stawka_runda2_trojka_bid_2minimum() {
    if (rezultat.poziom>2)
      return 2*gra.minimal_bid;
    else 
      return 0.0;    
  }

  private double stawka_runda2_trojka_bid_10minimum() {
    if (rezultat.poziom>2)
      return 10*gra.minimal_bid;
    else 
      return 0.0;    
  }    
  private double stawka_runda2_street_bid_10minimum() {
    if (rezultat.poziom>3)
      return 10*gra.minimal_bid;
    else 
      return 0.0;    
  }      
  
  private double stawka_runda3() {
    double propozycja = 0;
    if (individual.getGene(66) == 1)
      propozycja += stawka_runda2_para_bid_minimum();
    if (individual.getGene(67) == 1)
      propozycja += stawka_runda2_para_bid_5minimum();    
    if (individual.getGene(68) == 1)
      propozycja += stawka_runda2_2pary_bid_minimum();      
    if (individual.getGene(69) == 1)
      propozycja += stawka_runda2_2pary_bid_5minimum();       
    if (individual.getGene(70) == 1)
      propozycja += stawka_runda2_trojka_bid_2minimum();      
    if (individual.getGene(71) == 1)
      propozycja += stawka_runda2_trojka_bid_10minimum();           
    if (individual.getGene(72) == 1)
      propozycja += stawka_runda2_street_bid_10minimum();           

    return propozycja;
  }    
  
  private double stawka_runda4() {
    double propozycja = 0;
    if (individual.getGene(94) == 1)
      propozycja += stawka_runda2_para_bid_minimum();
    if (individual.getGene(95) == 1)
      propozycja += stawka_runda2_para_bid_5minimum();    
    if (individual.getGene(96) == 1)
      propozycja += stawka_runda2_2pary_bid_minimum();      
    if (individual.getGene(97) == 1)
      propozycja += stawka_runda2_2pary_bid_5minimum();       
    if (individual.getGene(98) == 1)
      propozycja += stawka_runda2_trojka_bid_2minimum();      
    if (individual.getGene(99) == 1)
      propozycja += stawka_runda2_trojka_bid_10minimum();           
    if (individual.getGene(100) == 1)
      propozycja += stawka_runda2_street_bid_10minimum();           

    return propozycja;
  }    
  
  
  private double stawka_runda2() {
    double propozycja = 0;
    if (individual.getGene(36) == 1)
      propozycja += stawka_runda2_para_bid_minimum();
    if (individual.getGene(37) == 1)
      propozycja += stawka_runda2_para_bid_5minimum();    
    if (individual.getGene(38) == 1)
      propozycja += stawka_runda2_2pary_bid_minimum();      
    if (individual.getGene(39) == 1)
      propozycja += stawka_runda2_2pary_bid_5minimum();       
    if (individual.getGene(40) == 1)
      propozycja += stawka_runda2_trojka_bid_2minimum();      
    if (individual.getGene(41) == 1)
      propozycja += stawka_runda2_trojka_bid_10minimum();           
    if (individual.getGene(42) == 1)
      propozycja += stawka_runda2_street_bid_10minimum();           

    return propozycja;
  }  
  
  
  

  private boolean czy_grac_2runda_zawsze_grac() {
    return true;
  }
  
  private boolean czy_grac_2runda_jest_para() {
    return rezultat.poziom >=1;
  }  
  
  private boolean czy_grac_2runda_jest_2pary() {
    return rezultat.poziom >=2;
  }    
  private boolean czy_grac_2runda_jest_trojka() {
    return rezultat.poziom >=3;
  }     
  private boolean czy_grac_2runda_jest_street() {
    return rezultat.poziom >=4;
  }     
  private boolean czy_grac_2runda_jest_4gracz() {
    return gra.graczy_w_grze>=4;
  }      
  private boolean czy_grac_2runda_jest_conajwyzej_3graczy() {
    return gra.graczy_w_grze<4;
  }     
  
  boolean czy_grac_runda2_tylko_stawka_minimum() {
    if (gra.stawka == gra.minimal_bid)
      return true;
    else 
      return false;
  }    
  
  boolean czy_grac_runda2_co_najwyzej_2minimum() {
    if (gra.stawka < 2 * gra.minimal_bid)
      return true;
    else 
      return false;
  }     
  
  boolean czy_grac_runda2_co_najwyzej_5minimum() {
    if (gra.stawka < 5 * gra.minimal_bid)
      return true;
    else 
      return false;
  }     
  
  boolean czy_grac_runda2_co_najwyzej_10minimum() {
    if (gra.stawka < 10 * gra.minimal_bid)
      return true;
    else 
      return false;
  }       
  
  boolean czy_grac_runda2_co_najmniej10() {
    if (gra.stawka >= 10 * gra.minimal_bid)
      return true;
    else 
      return false;
  }     
  
  boolean czy_grac_runda2_tak_gdy_jest_glos(int glosow) {
    if (glosow >= 1)
      return true;
    else 
      return false;
  }         
  
  boolean czy_grac_runda2_tak_gdy_sa_3glosy(int glosow) {
    if (glosow >= 3)
      return true;
    else 
      return false;
  }         
  
  boolean czy_grac_runda2_tak_gdy_sa_7glosy(int glosow) {
    if (glosow >= 7)
      return true;
    else 
      return false;  
  }
  
  
  private boolean czy_grac_runda2() {
    int glosow = 0;
    
    if (individual.getGene(21) == 1 && czy_grac_2runda_zawsze_grac())
      glosow++;        
    if (individual.getGene(22) == 1 && czy_grac_2runda_jest_para())
      glosow++;         
    if (individual.getGene(23) == 1 && czy_grac_2runda_jest_2pary())
      glosow++;         
    if (individual.getGene(24) == 1 && czy_grac_2runda_jest_trojka())
      glosow++;            
    if (individual.getGene(25) == 1 && czy_grac_2runda_jest_street())
      glosow++;          
    if (individual.getGene(26) == 1 && czy_grac_2runda_jest_4gracz())
      glosow++;           
    if (individual.getGene(27) == 1 && czy_grac_2runda_jest_conajwyzej_3graczy())
      glosow++;        
    if (individual.getGene(28) == 1 && czy_grac_runda2_tylko_stawka_minimum())
      glosow++;        
    if (individual.getGene(29) == 1 && czy_grac_runda2_co_najwyzej_2minimum())
      glosow++;       
    if (individual.getGene(30) == 1 && czy_grac_runda2_co_najwyzej_5minimum())
      glosow++;       
    if (individual.getGene(31) == 1 && czy_grac_runda2_co_najwyzej_10minimum())
      glosow++;     
    if (individual.getGene(32) == 1 && czy_grac_runda2_co_najmniej10())
      glosow++;         
    
    if (individual.getGene(47) == 1 && czy_pass_runda2_tak_gdy_jest_smiec())
      return true;    
    if (individual.getGene(48) == 1 && czy_pass_runda2_tak_gdy_jest_para())
      return true;      
    
    if (individual.getGene(33) == 1 && czy_grac_runda2_tak_gdy_jest_glos(glosow))
      return true;
    if (individual.getGene(34) == 1 && czy_grac_runda2_tak_gdy_sa_3glosy(glosow))
      return true;
    if (individual.getGene(35) == 1 && czy_grac_runda2_tak_gdy_sa_7glosy(glosow))
      return true;    
    
    return false;
  }
  
  
  
  private boolean czy_pass_runda2_tak_gdy_jest_para() {
    return rezultat.poziom <= 1 ;
  }

  private boolean czy_pass_runda2_tak_gdy_jest_smiec() {
    return rezultat.poziom == 0 ;
  }

  private boolean dobic_do_stawki(double propozycja) {
    if (individual.getGene(18) == 1 && czy_dobic_runda1_zawsze())
      return true;
    if (individual.getGene(19) == 1 && czy_dobic_runda1_gry_brakuje_cwiary(propozycja, gra.stawka))
      return true;
    if (individual.getGene(20) == 1 && czy_dobic_runda1_gry_brakuje_polowy(propozycja, gra.stawka))
      return true;    
    return false;    
    
  }
  
  public boolean czy_dobic_runda1_zawsze() {
    return true;
  }
  
  public boolean czy_dobic_runda1_gry_brakuje_cwiary(double propozycja, double stawka) {
    if (propozycja / stawka > 0.75)
      return true;
    else
      return false;
  } 
  
  public boolean czy_dobic_runda1_gry_brakuje_polowy(double propozycja, double stawka) {
    if (propozycja / stawka > 0.5)
      return true;
    else
      return false;
  }   
  

  private double stawka_runda1() {
    double propozycja = 0;
    if (individual.getGene(10) == 1)
      propozycja += stawka_runda1_2pary_bid_minimum();
    if (individual.getGene(11) == 1)
      propozycja += stawka_runda1_2pary_bid_3minimum();    
    if (individual.getGene(12) == 1)
      propozycja += stawka_runda1_2pary_bid_10minimum();       
    if (individual.getGene(13) == 1)
      propozycja += stawka_runda1_2kolory_bid_minimum();        
    if (individual.getGene(14) == 1)
      propozycja += stawka_runda1_2kolory_bid_3minimum();       
    if (individual.getGene(15) == 1)
      propozycja += stawka_runda1_2kolory_bid_10minimum();   
    if (individual.getGene(16) == 1)
      propozycja += stawka_runda1_wysoka_karta_minimum();     
    if (individual.getGene(17) == 1)
      propozycja += stawka_runda1_bardzo_wysoka_karta_minimum();      
    return propozycja;
  }

  double stawka_runda1_2pary_bid_minimum() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra.getPrivateCard(kolejnosc, 1).wysokosc)
      return gra.minimal_bid;
    else 
      return 0.0;    
  }

  double stawka_runda1_2pary_bid_3minimum() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra.getPrivateCard(kolejnosc, 1).wysokosc)
      return 3*gra.minimal_bid;
    else 
      return 0.0;    
  }  
  
  double stawka_runda1_2pary_bid_10minimum() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra.getPrivateCard(kolejnosc, 1).wysokosc)
      return 10*gra.minimal_bid;
    else 
      return 0.0;    
  }    
  
  double stawka_runda1_2kolory_bid_minimum() {
    if (gra.getPrivateCard(kolejnosc, 0).kolor == gra.getPrivateCard(kolejnosc, 1).kolor)
      return gra.minimal_bid;
    else 
      return 0.0;    
  }    
  
  double stawka_runda1_2kolory_bid_3minimum() {
    if (gra.getPrivateCard(kolejnosc, 0).kolor == gra.getPrivateCard(kolejnosc, 1).kolor)
      return 3*gra.minimal_bid;
    else 
      return 0.0;    
  }      
  
  double stawka_runda1_2kolory_bid_10minimum() {
    if (gra.getPrivateCard(kolejnosc, 0).kolor == gra.getPrivateCard(kolejnosc, 1).kolor)
      return 10*gra.minimal_bid;
    else 
      return 0.0;    
  }       
  
  double stawka_runda1_wysoka_karta_minimum() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc > 10 && gra.getPrivateCard(kolejnosc, 1).wysokosc > 10)
      return gra.minimal_bid;
    else 
      return 0.0;    
  }     
  
  double stawka_runda1_bardzo_wysoka_karta_minimum() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc > 12 && gra.getPrivateCard(kolejnosc, 1).wysokosc > 12)
      return 2*gra.minimal_bid;
    else 
      return 0.0;    
  }     
  
  boolean czy_grac_runda1() {
    int glosow = 0;
    if (individual.getGene(0) == 1 && czy_grac_runda1_zawsze_grac())
      glosow++;
    if (individual.getGene(1) == 1 && czy_grac_runda1_2karty_tej_samej_wagi())
      glosow++; 
    if (individual.getGene(2) == 1 && czy_grac_runda1_2karty_tego_samego_koloru())
      glosow++;     
    if (individual.getGene(3) == 1 && czy_grac_runda1_tylko_stawka_minimum())
      glosow++;       
    if (individual.getGene(4) == 1 && czy_grac_runda1_co_najwyzej_2minimum())
      glosow++;   
    if (individual.getGene(5) == 1 && czy_grac_runda1_co_najwyzej_5minimum())
      glosow++;        
    if (individual.getGene(6) == 1 && czy_grac_runda1_co_najwyzej_10minimum())
      glosow++;         

    if (individual.getGene(7) == 1 && czy_grac_runda1_tak_gdy_jest_glos(glosow))
      return true;
    if (individual.getGene(8) == 1 && czy_grac_runda1_tak_gdy_sa_2glosy(glosow))
      return true;
    if (individual.getGene(9) == 1 && czy_grac_runda1_tak_gdy_sa_5glosy(glosow))
      return true;    
    
    return false;
  }
  
  
  
  boolean czy_grac_runda1_zawsze_grac() {
    return true;
  }
  
  boolean czy_grac_runda1_2karty_tej_samej_wagi() {
    if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra.getPrivateCard(kolejnosc, 1).wysokosc)
      return true;
    else 
      return false;
  }
 
  boolean czy_grac_runda1_2karty_tego_samego_koloru() {
    if (gra.getPrivateCard(kolejnosc, 0).kolor == gra.getPrivateCard(kolejnosc, 1).kolor)
      return true;
    else 
      return false;
  }  
  
  boolean czy_grac_runda1_tylko_stawka_minimum() {
    if (gra.stawka == gra.minimal_bid)
      return true;
    else 
      return false;
  }    
  
  boolean czy_grac_runda1_co_najwyzej_2minimum() {
    if (gra.stawka < 2 * gra.minimal_bid)
      return true;
    else 
      return false;
  }     
  
  boolean czy_grac_runda1_co_najwyzej_5minimum() {
    if (gra.stawka < 5 * gra.minimal_bid)
      return true;
    else 
      return false;
  }     
  
  boolean czy_grac_runda1_co_najwyzej_10minimum() {
    if (gra.stawka < 10 * gra.minimal_bid)
      return true;
    else 
      return false;
  }       
 
  boolean czy_grac_runda1_tak_gdy_jest_glos(int glosow) {
    if (glosow >= 1)
      return true;
    else 
      return false;
  }         
  
  boolean czy_grac_runda1_tak_gdy_sa_2glosy(int glosow) {
    if (glosow >= 2)
      return true;
    else 
      return false;
  }         
  
  boolean czy_grac_runda1_tak_gdy_sa_5glosy(int glosow) {
    if (glosow >= 5)
      return true;
    else 
      return false;
  }    
  
}

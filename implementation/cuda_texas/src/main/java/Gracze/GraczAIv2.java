package Gracze;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.RegulaAbstrakcyjnaDobijania;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;
import engine.rezultaty.Rezultat;

public class GraczAIv2 extends Gracz {

  // lista regul wymaganych do wejscia;
  static RegulaAbstrakcyjna[][] regulyNaWejscie = null; // w pierwszym elemencie jest regula, ktora mowi ile musi byc glosow
  static RegulaAbstrakcyjna[][] regulyStawkaRunda = null;
  static RegulaAbstrakcyjnaDobijania[][] regulyDobijanie = null;
  
  public EvBinaryVectorIndividual individual = null;
  private int kolejnosc=0;  

  
  public GraczAIv2(EvBinaryVectorIndividual individual_, int kolejnosc_) {
    individual = individual_;
    kolejnosc = kolejnosc_;
    
    if (regulyNaWejscie == null) {
    
    regulyStawkaRunda = new RegulaAbstrakcyjna[4][];
    regulyNaWejscie = new RegulaAbstrakcyjna[4][];
    regulyDobijanie = new RegulaAbstrakcyjnaDobijania[4][];
    
    regulyStawkaRunda[1 - 1] = GeneratorRegul.generujRegulyStawkaRunda1(individual).toArray(new RegulaAbstrakcyjna[0]);
    regulyStawkaRunda[2 - 1] = GeneratorRegul.generujRegulyStawkaRundyKolejne(individual, 1).toArray(new RegulaAbstrakcyjna[0]);
    regulyStawkaRunda[3 - 1] = GeneratorRegul.generujRegulyStawkaRundyKolejne(individual, 2).toArray(new RegulaAbstrakcyjna[0]);
    regulyStawkaRunda[4 - 1] = GeneratorRegul.generujRegulyStawkaRundyKolejne(individual, 3).toArray(new RegulaAbstrakcyjna[0]);
    
    regulyDobijanie[1 - 1] = GeneratorRegul.generujRegulyDobijanieRunda1(individual).toArray(new RegulaAbstrakcyjnaDobijania[0]);
    regulyDobijanie[2 - 1] = GeneratorRegul.generujRegulyDobijanieRundyKolejne(individual, 1).toArray(new RegulaAbstrakcyjnaDobijania[0]);
    regulyDobijanie[3 - 1] = GeneratorRegul.generujRegulyDobijanieRundyKolejne(individual, 2).toArray(new RegulaAbstrakcyjnaDobijania[0]);
    regulyDobijanie[4 - 1] = GeneratorRegul.generujRegulyDobijanieRundyKolejne(individual, 3).toArray(new RegulaAbstrakcyjnaDobijania[0]);
    
    regulyNaWejscie[1 - 1] = GeneratorRegul.generujRegulyNaWejscie(individual).toArray(new RegulaAbstrakcyjna[0]);
    regulyNaWejscie[2 - 1] = GeneratorRegul.generujRegulyNaWejscieRundyKolejne(individual, 1).toArray(new RegulaAbstrakcyjna[0]);
    regulyNaWejscie[3 - 1] = GeneratorRegul.generujRegulyNaWejscieRundyKolejne(individual, 2).toArray(new RegulaAbstrakcyjna[0]);
    regulyNaWejscie[4 - 1] = GeneratorRegul.generujRegulyNaWejscieRundyKolejne(individual, 3).toArray(new RegulaAbstrakcyjna[0]);
    
    }
  }
  
  @Override
  final public double play(final int i, final double bid) {

    if (i!=1)
      rezultat = Rezultat.pobierzPrognoze(gra, kolejnosc);
    
    if (!rundaX_czy_grac(i)) {
      return -1.0d;
    }
    else {

      double stawka = rundaX_stawka(i);
      
      if (stawka < gra.stawka) {
        if (rundaX_dobijanie(stawka, i))
          stawka = gra.stawka;
        else {
          return -1.0d;
        }
      }
      
      bilans -= stawka - bid;  
      if (musik > 0) {
        bilans +=musik;
        musik = 0;
      }

      return stawka;
    }          
    
  }

  
   private Rezultat rezultat = null; 
  
  static RegulaAbstrakcyjna[] tablicaRegul = null;
  static RegulaAbstrakcyjnaDobijania[] tablicaRegulDobijania = null;
   
  // true, jesli reguly stwierdzily, zeby wziac udzial w licytacji w 1 rundzie
  final public boolean rundaX_czy_grac(final int aRunda) {
    
    int pGlosow = 0;

    tablicaRegul = regulyNaWejscie[aRunda - 1];
    
    final int pWymaganychGlosow = (int)tablicaRegul[0].aplikujRegule(gra, kolejnosc, individual,rezultat);
    
    final int pLiczbaRegul = tablicaRegul.length;
    
    for (int i=1; i < pLiczbaRegul; i++) {
      pGlosow += tablicaRegul[i].aplikujRegule(gra, kolejnosc, individual,rezultat);
    }
       
    // ostateczna decyzja
    if (pGlosow <= pWymaganychGlosow)
      return true;
    else return false;    
    
  } 
  
  final public double rundaX_stawka(final int aRunda) {
   
    double pStawka = 0.0d;
    
    tablicaRegul = regulyStawkaRunda[aRunda - 1];
    
    final int pLiczbaRegul = tablicaRegul.length;
    
    for (int i=0; i < pLiczbaRegul; i++) {
      pStawka += gra.minimal_bid * tablicaRegul[i].aplikujRegule(gra, kolejnosc, individual,rezultat);
    }
    
    return pStawka;
  }
  
  
  
  final public boolean rundaX_dobijanie(final double aStawka, final int aRunda) {
  
    tablicaRegulDobijania = regulyDobijanie[aRunda - 1];
    
    final int pLiczbaRegul = tablicaRegulDobijania.length;
    
    for (int i=0; i < pLiczbaRegul; i++) {    
    
      if ( tablicaRegulDobijania[i].aplikujRegule(gra, kolejnosc, aStawka, individual, rezultat) == 1.0d )
        return true;
      
    }

    return false;
  }
  
  
  
//  public String toString() {
//    
//    String ret = new String();
//    
//    Runda1_CzyGrac runda1_czy_grac = new Runda1_CzyGrac(gra, kolejnosc, individual, CZ_GRAC_RUNDA1);
//    Runda1_Stawka runda1_stawka = new Runda1_Stawka(gra, kolejnosc, individual, STAWKA_RUNDA1);
//    Runda1_Dobijanie runda1_dobijanie = new Runda1_Dobijanie(gra, kolejnosc, individual, STAWKA_RUNDA1);
//   
//    Runda2_CzyGrac runda2_czy_grac = new Runda2_CzyGrac(gra, kolejnosc, individual, CZ_GRAC_RUNDA2);
//    Runda2_Stawka runda2_stawka = new Runda2_Stawka(gra, kolejnosc, individual, STAWKA_RUNDA2);
//    Runda2_Dobijanie runda2_dobijanie = new Runda2_Dobijanie(gra, kolejnosc, individual, STAWKA_RUNDA2);    
//    
//    ret = runda1_czy_grac + "\n" + runda1_stawka + "\n" + runda1_dobijanie +"\n";
//    ret += runda2_czy_grac + "\n" + runda2_stawka + "\n" + runda2_dobijanie +"\n";
//    return ret;
//  }
  
  
}

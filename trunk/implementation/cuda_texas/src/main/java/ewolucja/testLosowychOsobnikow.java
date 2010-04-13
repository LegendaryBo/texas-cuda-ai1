package ewolucja;

import generator.IndividualGenerator;

import java.util.Arrays;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

/**
 * 
 * Test ma na celu sprawdzic jak dobrego osobnika da sie wygenerowac przy pomocy duzej ilosci losowan
 * 
 * @author Kacper Gorski
 *
 */
public class testLosowychOsobnikow {

 public static void main(String[] args) {
    
    GeneratorRegul.init();
    
    TexasObjectiveFunction objective_function = new TexasObjectiveFunction(5000, 10 , true, false);

    final int genes = GeneratorRegul.rozmiarGenomu;
    final int LOSOWAN = 100000;
  
    IndividualGenerator generatorOsobnikow = new IndividualGenerator(124, genes);
    
    double pMax = Double.MIN_VALUE;
    double pMin = Double.MAX_VALUE;
    
    double[] tab = new double[LOSOWAN];
    double suma = 0.0d;
    
    for (int i=0; i < LOSOWAN; i++) {
      EvBinaryVectorIndividual osobnik = generatorOsobnikow.generate();
      osobnik.setObjectiveFunction(objective_function);
      double pWartoscFunkcjiCelu = osobnik.getObjectiveFunctionValue();
      
      tab[i] = pWartoscFunkcjiCelu;
      suma+=tab[i];
      
      if (pWartoscFunkcjiCelu>pMax)
        pMax = pWartoscFunkcjiCelu;
      
      if (pWartoscFunkcjiCelu<pMin)
        pMin = pWartoscFunkcjiCelu;      
      
      

      
      System.out.println("Osobnik "+i+" wartosc funkcji celu: "+pWartoscFunkcjiCelu);
    }
    
    suma = suma/LOSOWAN;
    double stdDev = 0.0d;
    
    for (int i=0; i < LOSOWAN; i++) {
      double dev = Math.abs(suma - tab[i]);
      stdDev += dev;
    }
    
    stdDev = stdDev/LOSOWAN;    
    
    double dolne10Proc = 0.0d;
    double dolne1Proc = 0.0d;
    double dolne01Proc = 0.0d;
    
    Arrays.sort(tab);
    for (int i=0; i < LOSOWAN/10; i++)
      dolne10Proc += tab[LOSOWAN - 1 - i]/(LOSOWAN/10);
    
    for (int i=0; i < LOSOWAN/100; i++)
      dolne1Proc += tab[LOSOWAN - 1 - i]/(LOSOWAN/100);
    
    for (int i=0; i < LOSOWAN/1000; i++)
      dolne01Proc += tab[LOSOWAN - 1 - i]/(LOSOWAN/1000);
  
    System.out.println("Najlepszy osobnik: "+pMax);
    System.out.println("Najgorszy osobnik: "+pMin);
    System.out.println("Srednia wartosc funkcji celu: "+suma);
    System.out.println("Srednie odchylenie standardowe: "+stdDev+" "+Math.round(Math.abs(stdDev/suma)*10000)/100.0f+"%");
    System.out.println("Srednia wartosc najlepszych 10% osobnikow: "+dolne10Proc+"");
    System.out.println("Srednia wartosc najlepszych 1% osobnikow: "+dolne1Proc+"");
    System.out.println("Srednia wartosc najlepszych 0.1% osobnikow: "+dolne01Proc+"");
 }
  
}

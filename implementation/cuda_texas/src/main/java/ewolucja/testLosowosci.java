package ewolucja;

import generator.IndividualGenerator;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;
import Gracze.gracz_v3.GeneratorRegulv3;

public class testLosowosci {

  /**
   * @param args
   */
  public static void main(String[] args) {
    
    GeneratorRegul.init();
    
//    GeneratorGraczyZGeneracji generator = 
//      new GeneratorGraczyZGeneracji(11314, GeneratorRegulv3.rozmiarGenomu, 0, true);  

    IndividualGenerator generator = new IndividualGenerator(44, GeneratorRegulv3.rozmiarGenomu);
    System.out.println(GeneratorRegulv3.rozmiarGenomu);
    EvBinaryVectorIndividual individual = generator.generate();
    
    TexasObjectiveFunction objective_function = new TexasObjectiveFunction(3000, 0, true, false);   
    
    double stdDev = 0.0d;
    double suma = 0.0d;
    
    final int OSOBNIKOW = 500;
    
    double[] tab = new double[OSOBNIKOW];
    
    for (int i=0; i < OSOBNIKOW; i++) {
      individual.setObjectiveFunction(objective_function);
      tab[i] = individual.getObjectiveFunctionValue();
      System.out.println("Osobnik "+i+", wartosc funkcji celu "+tab[i]);
      suma+=tab[i];
    }
    suma = suma/OSOBNIKOW;
    
    for (int i=0; i < OSOBNIKOW; i++) {
      double dev = Math.abs(suma - tab[i]);
      stdDev += dev;
    }
    stdDev = stdDev/OSOBNIKOW;
    
    System.out.println("Srednia wartosc funkcji celu: "+suma);
    System.out.println("Srednie odchylenie standardowe: "+stdDev+" "+Math.round(Math.abs(stdDev/suma)*10000)/100.0f+"%");

  }

}

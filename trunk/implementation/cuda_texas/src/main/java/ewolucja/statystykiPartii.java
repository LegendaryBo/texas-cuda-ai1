package ewolucja;

import generator.GeneratorGraczyZGeneracji;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

public final class statystykiPartii {

  /**
   * @param args
   */
  public static void main(String[] args) {
    
    final int SPRAWDZANA_GENERACJA = 8;
    final int GENERACJA_FUNKCJI_CELU = 4;
    final int ITERACJI = 20;
    
    GeneratorRegul.init();
    
    GeneratorGraczyZGeneracji generator = 
      new GeneratorGraczyZGeneracji(1144, GeneratorRegul.rozmiarGenomu, SPRAWDZANA_GENERACJA, true);  
    TexasObjectiveFunction objective_function = new TexasObjectiveFunction(2000, GENERACJA_FUNKCJI_CELU, true, false);   
    
    EvBinaryVectorIndividual individual = null;
    
    double srednia = 0.0d;
    
    for (int i=0; i < ITERACJI; i++) {
        
      individual = generator.generate();
      individual.setObjectiveFunction(objective_function);
      double pObjValue = individual.getObjectiveFunctionValue();
      System.out.println("Osobnik "+ (i+1) +" "+pObjValue);
      System.out.println("statystyki "+ objective_function.getStats(individual));
      System.out.println();
      srednia += pObjValue;
    }

    
    System.out.println("Sredni osobnik"+(srednia/ITERACJI));
  }

}

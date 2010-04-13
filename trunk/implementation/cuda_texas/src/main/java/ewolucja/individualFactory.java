package ewolucja;

import generator.IndividualIO;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

public class individualFactory {

  /**
   * @param args
   */
  public static void main(String[] args) {
    
    TexasObjectiveFunction objective_function = new TexasObjectiveFunction(100000, 7, true, true);

    EvPopulation population = new EvPopulation( IndividualIO.odczytajZPliku("c:\\texas3\\generacja7.dat"));
    
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual) population.get(0);
    best.setObjectiveFunction(objective_function);
    
    System.out.println("wartosc najlepszego osobnika przed poprawka"+best.getObjectiveFunctionValue());
    
    //GeneratorRegul.regulaWysokieKartyStawkaR1.zmienIndividuala(new double[]{ 1.0d, 512.0d }, best);
    System.out.println(GeneratorRegul.regulaCzyParaWReceStawkaR1.kodGraya.getWartoscKoduGraya(best));

    GeneratorRegul.regulaCzyParaWReceStawkaR1.zmienIndividuala(new double[]{ 1.0d, 140.0d }, best);
    System.out.println(GeneratorRegul.regulaCzyParaWReceStawkaR1.kodGraya.getWartoscKoduGraya(best));

    
    System.out.println("wartosc najlepszego osobnika po poprawce"+best.getObjectiveFunctionValue());
  }

}

package ewolucja.programy;

import generator.IndividualIO;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

public class RecznePoprawieniaOsobnika {

  /**
   * Ten program oblicza podany osobnik, a nastepnie wyswietla wynik dla osobnika, w ktorym recznie zmienilismy jeden gen
   * 
   * @param args
   */
  @SuppressWarnings("unchecked")
public static void main(String[] args) {
    
    TexasObjectiveFunction objective_function = new TexasObjectiveFunction(100000, 7, true, true);

    EvPopulation population = new EvPopulation( IndividualIO.odczytajZPliku(
    "/home/railman/workspace/texas/implementation/cuda_texas/src/main/resources/texas_individuale/generacja7.dat"));
    
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual) population.get(0);
    best.setObjectiveFunction(objective_function);
    
    System.out.println("wartosc najlepszego osobnika przed poprawka "+best.getObjectiveFunctionValue());
    
    //GeneratorRegul.regulaWysokieKartyStawkaR1.zmienIndividuala(new double[]{ 1.0d, 512.0d }, best);
    System.out.println(GeneratorRegul.regulaCzyParaWReceStawkaR1.kodGraya.getWartoscKoduGraya(best));

    GeneratorRegul.regulaCzyParaWReceStawkaR1.zmienIndividuala(new double[]{ 1.0d, 140.0d }, best);
    System.out.println(GeneratorRegul.regulaCzyParaWReceStawkaR1.kodGraya.getWartoscKoduGraya(best));

    
    System.out.println("wartosc najlepszego osobnika po poprawce"+best.getObjectiveFunctionValue());
  }

}

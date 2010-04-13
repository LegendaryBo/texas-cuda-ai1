package testy.wevo;

import generator.GeneratorGraczyZGeneracji;
import generator.IndividualIO;

import java.util.ArrayList;
import java.util.Date;

import operatory.WywalDuplikaty;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

public class TestIndividuali {

  /**
   * @param args
   */
  public static void main(String[] args) {
    
    GeneratorRegul.init();
    
    final int GENES = GeneratorRegul.rozmiarGenomu;
    
    TexasObjectiveFunction fun = new TexasObjectiveFunction(5000,1, false, false);
    
    //GeneratorBardzoDobrychGraczy generator = new GeneratorBardzoDobrychGraczy(6666);
    //GeneratorDobrychGraczy generator = new GeneratorDobrychGraczy(6666);
    GeneratorGraczyZGeneracji generatorGraczy = new GeneratorGraczyZGeneracji(55312, GENES, 0, true);
    EvBinaryVectorIndividual ind = generatorGraczy.generate();
    ind.setObjectiveFunction(fun);
    
    long pStartTime = (new Date()).getTime();
    
    System.out.println("Wartosc funkcji celu: "+ind.getObjectiveFunctionValue());
    

    
    ArrayList<EvBinaryVectorIndividual> aList = IndividualIO.odczytajZPliku("c:\\texas3\\generacja1.dat");
    
    System.out.println("bylo: "+aList.size());
    WywalDuplikaty.aplikuj(aList);

    double srednia = 0.0d;
    
    for (EvBinaryVectorIndividual evBinaryVectorIndividual : aList) {
      evBinaryVectorIndividual.setObjectiveFunction(fun);
      srednia +=evBinaryVectorIndividual.getObjectiveFunctionValue();
      System.out.println(evBinaryVectorIndividual.getObjectiveFunctionValue());
    }
    
    System.out.println("srednia: "+srednia/aList.size());
    
    aList = WywalDuplikaty.selekcjaDoPliku(aList);
    System.out.println("jest: "+aList.size());

    long pStopTime = (new Date()).getTime();
    
    System.out.println("Czas obliczania jednej funkcji celu: "+(pStopTime - pStartTime));  
    
  }

}

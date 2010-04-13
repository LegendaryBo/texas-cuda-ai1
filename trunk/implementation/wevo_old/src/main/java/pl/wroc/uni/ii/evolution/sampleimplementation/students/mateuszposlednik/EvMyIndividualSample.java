/**
 * 
 */
package pl.wroc.uni.ii.evolution.sampleimplementation.students.mateuszposlednik;

// off LineLength
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;

// on LineLength

/**
 * Przyklad uzycia MyIndividual wraz z operatorem i funkcja celu.
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public final class EvMyIndividualSample {

  /** wielkosc populacji. */
  static final int POPULATIONSIZE = 100;

  /** dlugosc chromosonu. */
  static final int CHROMOSONELENGHT = 20;

  /** ilosc iteracji. */
  static final int ITERATIONS = 10;


  /** konstruktor. */
  private EvMyIndividualSample() {

  }


  /**
   * Wywolanie programu.
   * 
   * @param args nie uzywane
   */
  public static void main(final String[] args) {

    // tworzymy funkcje celu:
    EvMyIndividualObjective objective = new EvMyIndividualObjective();
    // tworzymy algorytm:
    EvAlgorithm<EvMyIndividual> alg =
        new EvAlgorithm<EvMyIndividual>(EvMyIndividualSample.POPULATIONSIZE);
    // tworzymy przestrzen poszukiwan:
    EvMyIndividualSpace space =
        new EvMyIndividualSpace(objective,
            EvMyIndividualSample.CHROMOSONELENGHT);
    // tworzymy operator:
    EvMyIndividualMutationOperator mutation =
        new EvMyIndividualMutationOperator();
    // tworzymy warunek zakonczenia:
    EvMaxIteration<EvMyIndividual> termination =
        new EvMaxIteration<EvMyIndividual>(EvMyIndividualSample.ITERATIONS);
    // tworzymy dodatkowy operator co bedzie wyswietlal kolejne iteracje:
    EvRealtimeToPrintStreamStatistics<EvMyIndividual> show_operator =
        new EvRealtimeToPrintStreamStatistics<EvMyIndividual>(System.out);

    // skladamy wszystko w calosc:
    alg.setObjectiveFunction(objective);
    alg.setSolutionSpace(space);
    alg.setTerminationCondition(termination);
    alg.addOperator(mutation);
    alg.addOperator(show_operator);

    // inicjujemy i odpalamy algorytm:
    // alg.init();
    // alg.run();

    // wersja z taskiem:
    EvTask evolutionary_task = new EvTask();
    evolutionary_task.setAlgorithm(alg);
    evolutionary_task.run();

    // drukujemy najlepszy osobnik:
    // EvMyIndividual the_best = alg.getBestResult();
    // System.out.println("najlepszy: " + the_best);

    evolutionary_task.printBestResult();
  }

}

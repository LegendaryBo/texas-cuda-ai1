package pl.wroc.uni.ii.evolution.engine.operators.general.display;

import java.io.PrintStream;
import java.util.Calendar;
import java.util.GregorianCalendar;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Display at console simple statistic about population
 * 
 * @author Marcin Golebiowski
 */

public class EvRealtimeToPrintStreamStatistics<T extends EvIndividual>
    implements EvOperator<T> {

  private int generation_number = 0;

  private PrintStream print_stream = null;


  /**
   * Constructor.
   * 
   * @param stream -- to this stream basic stats will be written
   */
  public EvRealtimeToPrintStreamStatistics(PrintStream stream) {
    print_stream = stream;
  }


  public EvPopulation<T> apply(EvPopulation<T> population) {

    Calendar cal = new GregorianCalendar();

    // Get the components of the time // 0..11
    int hour24 = cal.get(Calendar.HOUR_OF_DAY); // 0..23
    int min = cal.get(Calendar.MINUTE); // 0..59
    int sec = cal.get(Calendar.SECOND); // 0..59

    print_stream.println("----");
    print_stream.println(hour24 + ":" + min + ":" + sec);
    print_stream.println("Generation: " + generation_number);
    generation_number++;

    double sum = 0.0;

    for (int i = 0; i < population.size(); i++) {
      sum += population.get(i).getObjectiveFunctionValue();
    }

    print_stream.println("Population size: " + population.size());
    print_stream.println("Best individual: " + population.getBestResult());
    print_stream.println("Worst individual: " + population.getWorstResult());
    print_stream.println("Best fitness: "
        + population.getBestResult().getObjectiveFunctionValue());
    print_stream
        .println("Avarage fitness: " + sum / (double) population.size());
    print_stream.println("Worst fitness: "
        + population.getWorstResult().getObjectiveFunctionValue());
    print_stream.println("-----------------");
    return population;
  }
}
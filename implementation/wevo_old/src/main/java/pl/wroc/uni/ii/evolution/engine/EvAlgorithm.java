package pl.wroc.uni.ii.evolution.engine;

import java.util.LinkedList;

import pl.wroc.uni.ii.evolution.distribution.workers.EvBlankEvolInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.engine.prototype.conditions.EvTerminationCondition;

/**
 * Generic evolutionary algorithm that can be customized with an apropriate
 * objective function and evolutionary operators. This is the most general
 * implementation we could devise, if you can further generalize it, feel free
 * to do so.
 * 
 * @author Marcin Brodziak, Marcin Golebiowski
 */
public class EvAlgorithm<T extends EvIndividual> {

  protected EvPopulation<T> population;

  protected EvTerminationCondition<T> termination_condition;

  protected LinkedList<EvOperator<T>> operators;

  protected EvOperator<T> first_operator;

  protected EvOperator<T> last_operator;

  protected EvSolutionSpace<T> solution_space;

  protected EvObjectiveFunction<T> objective_function;

  protected int population_size;

  protected int iteration_number;

  private EvEvolutionInterface inter;


  public EvAlgorithm(int population_size) {
    if (population_size <= 0) {
      throw new IllegalArgumentException(
          "Population size must be positive integer");
    }

    this.population_size = population_size;
    this.operators = new LinkedList<EvOperator<T>>();
  }


  /**
   * Standard method for every EvolutionaryAlgorithm. Initializes population.
   * Checks if solution space is null.<BR>
   * It also sets an objective function from solution space if it's not set.
   */
  public void init() {

    if (solution_space == null) {
      throw new IllegalStateException("Solution space is not set!");
    }

    // if objective_function wasn't set manually,
    // we take an objective function from solution space by default
    if (objective_function == null) {
      setObjectiveFunction(solution_space.getObjectiveFuntion());
      if (objective_function == null)
        throw new IllegalStateException("Objective function is not set!");
    }

    population = new EvPopulation<T>();

    for (int i = 0; i < population_size; i++) {
      T generateIndividual = (T) solution_space.generateIndividual();
      generateIndividual.setObjectiveFunction(objective_function);
      population.add(generateIndividual);
    }
  }


  /**
   * Let's get the best result. Uses population method
   */
  public T getBestResult() {
    return population.getBestResult();
  }


  public void setLastOperator(EvOperator<T> op) {
    this.last_operator = op;
  }


  public void setFirstOperator(EvOperator<T> op) {
    this.first_operator = op;
  }


  /**
   * The worst individual in population.
   */
  public T getWorstResult() {
    return population.getWorstResult();
  }


  /**
   * Adds operator to the list of EvolutionaryOperators that are considered in
   * the source code. The order of adding new operators is equal with the order
   * of application on population in each iteration.
   * 
   * @param op -- operator to be added
   */
  public void addOperatorToEnd(EvOperator<T> op) {
    operators.addLast(op);
  }


  /**
   * Does exacly the same as addOperatorToEnd
   * 
   * @param op -- operator to be added
   */
  public void addOperator(EvOperator<T> op) {
    addOperatorToEnd(op);
  }


  /**
   * Add operators to the beggining of the list of evolutionary operators. All
   * present operators are pushed one place ahead.
   * 
   * @param op -- operator to be added
   */
  public void addOperatorToBeginning(EvOperator<T> op) {
    operators.addFirst(op);
  }


  /** Checks if algorithm reach the end. */
  public boolean isTerminationConditionSatisfied() {
    return termination_condition.isSatisfied();
  }


  /**
   * Sets Objective function used to evaluate individuals.<BR>
   * <BR>
   * NOTE: by default the objective function is taken from the solution space
   * kept in the algorithm, so there is no necesarity (if there is an objective
   * funtion set in a solution space) to call this function
   * 
   * @param function
   */
  public void setObjectiveFunction(EvObjectiveFunction<T> function) {
    this.objective_function = function;
  }


  /* Accessors, that are common for all subclasses */
  public void setSolutionSpace(EvSolutionSpace<T> space) {
    this.solution_space = space;
  }


  /**
   * Sets the condition under which the algorithm runs.
   * 
   * @param condition
   */
  public void setTerminationCondition(EvTerminationCondition<T> condition) {
    termination_condition = condition;
  }


  /**
   * Run algorithm till termination condition is satisfied.<BR>
   * Remember to set termination_condition before running this method.
   */
  public void run() {
    if (termination_condition == null)
      throw new IllegalStateException(
          "Termination condition is not set or se to null!");
    while (!termination_condition.isSatisfied()) {
      doIteration();
    }
  }


  /**
   * Return current population evaluated by the algorithm
   * 
   * @return
   */
  public EvPopulation<T> getPopulation() {
    return population;
  }


  public void setPopulation(EvPopulation<T> pop) {
    this.population = pop;
    this.population_size = pop.size();
  }


  /**
   * Standard method for doing single iteration. Just applies consequently every
   * operator to a population.
   */
  public void doIteration() {
    if (inter == null)
      inter = new EvBlankEvolInterface();

    if (operators.size() == 0) {
      throw new IllegalStateException(
          "There aren't any operators in the algorithm !");
    }

    if (termination_condition == null) {
      throw new IllegalStateException("Terminal condition is not set!");
    }

    /*
     * Iteration on Operators. Each operator is applied to the population.
     */

    if (first_operator != null) {
      // System.out.println("First apply");
      long start_time = System.currentTimeMillis();
      population = first_operator.apply(population);
      long end_time = System.currentTimeMillis();
      inter.addOperatortime(first_operator, (int) (end_time - start_time));

    }

    for (EvOperator<T> operator : operators) {

      // System.out.println("Apply " + operator.toString());
      long start_time = System.currentTimeMillis();
      population = operator.apply(population);
      long end_time = System.currentTimeMillis();
      inter.addOperatortime(operator, (int) (end_time - start_time));
      /*
       * Some very basic sanity check. Population cannot be null or empty.
       */
      if (population == null) {
        throw new IllegalStateException("Operator " + operator.getClass()
            + " has returned null instead of valid population");
      }

      if (population.size() == 0) {
        throw new IllegalStateException("Operator " + operator.getClass()
            + " has shrunk population to zero.");
      }
    }

    if (last_operator != null) {
      // System.out.println("Last apply");
      long start_time = System.currentTimeMillis();
      population = last_operator.apply(population);
      long end_time = System.currentTimeMillis();
      inter.addOperatortime(last_operator, (int) (end_time - start_time));
    }

    /*
     * Updating termination condition.
     */
    termination_condition.changeState(population);
  }


  /**
   * Sets the interface that sets details of what is happening here
   * 
   * @param inter
   */
  public void setInterface(EvEvolutionInterface inter) {
    this.inter = inter;
  }
}

package pl.wroc.uni.ii.evolution.experimental.geneticprogramming;

import java.util.GregorianCalendar;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPTree;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Zbigniew Nazimek, Donata Ma�ecka
 */
public class EvGPTreeCrossover implements EvOperator<EvGPTree> {

  EvPopulation<EvGPTree> children = new EvPopulation<EvGPTree>();

  EvRandomizer rand =
      new EvRandomizer(new GregorianCalendar().getTimeInMillis());


  public EvPopulation<EvGPTree> apply(EvPopulation<EvGPTree> population_) {
    children = new EvPopulation<EvGPTree>();
    EvPopulation<EvGPTree> population =
        (EvPopulation<EvGPTree>) population_.clone();
    if (population == null)
      return population;
    if (population.size() < 2)
      return population;

    if (population.size() % 2 == 1) {
      children.add((EvGPTree) population.get(0).clone());
      population.remove(0);
    }

    while (population.size() > 0) {
      int randNum = rand.nextInt(population.size());
      EvGPTree parent1 = population.get(randNum);
      population.remove(randNum);

      randNum = rand.nextInt(population.size());
      EvGPTree parent2 = population.get(randNum);
      population.remove(randNum);

      crossover(parent1, parent2);

    }

    return children;
  }


  private void crossover(EvGPTree parent1, EvGPTree parent2) {
    EvGPTree child1 = (EvGPTree) parent1.clone();
    EvGPTree child2 = (EvGPTree) parent1.clone();

    int height1 = parent1.getHeight();
    int height2 = parent2.getHeight();

    int r1 = rand.nextInt(height1);
    int r2 = rand.nextInt(height2);

    EvGPTree sub1 = child1;
    EvGPTree sub2 = child2;
    boolean choose1 = false;
    boolean choose2 = false;
    int level = 1;

    while ((!choose1) && (!choose2)) {
      if (level == r1)
        choose1 = true;
      else {
        if (rand.nextBoolean()) {
          if (sub1.hasLeft())
            sub1 = sub1.getLeftSubTree();
          else if (sub1.hasRight())
            sub1 = sub1.getRightSubTree();
          else
            choose1 = true;
        }
      }

      if (level == r2)
        choose2 = true;
      else {
        if (rand.nextBoolean()) {
          if (sub2.hasLeft())
            sub2 = sub2.getLeftSubTree();
          else if (sub2.hasRight())
            sub2 = sub2.getRightSubTree();
          else
            choose2 = true;
        }
      }

      level++;
    }

    EvGPTree tmp = (EvGPTree) sub1.clone();
    sub1 = (EvGPTree) sub2.clone();
    sub2 = (EvGPTree) tmp.clone();

    children.add(sub1);
    children.add(sub2);
  }

}

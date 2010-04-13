package pl.wroc.uni.ii.evolution.utils.benchmark;

import java.util.ArrayList;
import java.util.Random;

/**
 * <i>Shootout</i> benchmark. Benchmark meant to test system rating with
 * mutltithreaded job running. The <code>shootout</code> class represents
 * multithreaded algorithm, with separate threads (as instances of class
 * <code>Fighter</code>) eliminating each other.
 * 
 * @author Krzysztof Sroka
 */
public final class EvShootout {

  private int warriors_left;

  private Arena arena;

  // Fighter is a threaded object with associated
  // strength (double) and lives (initially 3).
  // Fihter runs until its the only one left or until
  // it runs out of lives.
  private class Fighter extends Thread {

    private int lives;

    private double strength;


    // constructor
    public Fighter(double strength) {
      lives = 3;
      this.strength = strength;
    }


    // runs the thread
    public void run() {
      while (lives > 0 && warriors_left > 1) {
        try {
          arena.fight(this);
          // let other threads enter
          synchronized (this) {
            wait(1);
          }
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
      }
      if (lives == 0) {
        synchronized (this) {
          warriors_left--;
        }
      }
    }
  }

  private class Arena {

    private Fighter first;


    public Arena() {
    }


    public synchronized void fight(Fighter f) throws InterruptedException {

      if (first == null) {
        first = f;
      } else if (f != first) {
        double str_first = first.strength;
        double str_second = f.strength;

        if (str_first < str_second) {
          first.lives--;
        } else if (str_first > str_second) {
          f.lives--;
        } else {
          if ((new Random()).nextBoolean()) {
            first.lives--;
          } else {
            f.lives--;
          }
        }

        first = null;
      }

    }
  }


  /**
   * @param fighters_size
   * @throws InterruptedException
   */
  public EvShootout(int fighters_size) throws InterruptedException {
    warriors_left = fighters_size;

    ArrayList<Fighter> warriors = new ArrayList<Fighter>(fighters_size);
    this.arena = new Arena();

    Random r = new Random();

    for (int i = 0; i < fighters_size; i++) {
      warriors.add(new Fighter(r.nextDouble()));
      warriors.get(i).start();
    }

    for (int i = 0; i < fighters_size; i++) {
      warriors.get(i).join();
    }

    if (warriors_left != 1) {
      System.out.println("WTF!!! " + warriors_left);
    }
  }


  /**
   * Run the algorithm. Creates threads, runs them, and waits for their finish.
   * 
   * @param fighters nmber of threads to be created
   * @return time spent between creating the first thread and joining the last
   *         one.
   */
  public static long battle(int fighters) {

    long time_start = System.currentTimeMillis();

    try {
      new EvShootout(fighters);
    } catch (InterruptedException e) {
      // finish
    }

    return System.currentTimeMillis() - time_start;
  }
}

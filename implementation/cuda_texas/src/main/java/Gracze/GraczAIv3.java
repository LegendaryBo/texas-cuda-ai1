package Gracze;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.RegulaAbstrakcyjnaDobijania;
import reguly.RegulaAbstrakcyjnaIleGrac;
import reguly.dobijanie.RegulaDobijajGdyBrakujeX;
import Gracze.gracz_v3.GeneratorRegulv3;
import engine.Gra;
import engine.rezultaty.Rezultat;

public class GraczAIv3 extends Gracz {

	public Rezultat rezultat = null;

	static RegulaAbstrakcyjna[][] regulyNaWejscie = null; // w pierwszym
								// elemencie
								// jest regula,
								// ktora mowi
								// ile musi byc
								// glosow
	static RegulaAbstrakcyjna[][] regulyStawkaRunda = null;
	static RegulaAbstrakcyjnaDobijania[][] regulyDobijanie = null;
	static RegulaAbstrakcyjnaIleGrac[][] regulyIleGrac = null;

	private int kolejnosc = 0;
	public EvBinaryVectorIndividual individual = null;

	public GraczAIv3(EvBinaryVectorIndividual individual_, int kolejnosc_) {
		individual = individual_;
		kolejnosc = kolejnosc_;

		if (regulyNaWejscie == null) {

			regulyStawkaRunda = new RegulaAbstrakcyjna[4][];
			regulyNaWejscie = new RegulaAbstrakcyjna[4][];
			regulyDobijanie = new RegulaAbstrakcyjnaDobijania[4][];
			regulyIleGrac = new RegulaAbstrakcyjnaIleGrac[4][];

			regulyStawkaRunda[1 - 1] = GeneratorRegulv3
					.generujRegulyStawkaRunda1(individual)
					.toArray(new RegulaAbstrakcyjna[0]);
			regulyStawkaRunda[2 - 1] = GeneratorRegulv3
					.generujRegulyStawkaRundyKolejne(
							individual, 1)
					.toArray(new RegulaAbstrakcyjna[0]);
			regulyStawkaRunda[3 - 1] = GeneratorRegulv3
					.generujRegulyStawkaRundyKolejne(
							individual, 2)
					.toArray(new RegulaAbstrakcyjna[0]);
			regulyStawkaRunda[4 - 1] = GeneratorRegulv3
					.generujRegulyStawkaRundyKolejne(
							individual, 3)
					.toArray(new RegulaAbstrakcyjna[0]);

			regulyDobijanie[1 - 1] = GeneratorRegulv3
					.generujRegulyDobijanieRunda1(
							individual)
					.toArray(
							new RegulaAbstrakcyjnaDobijania[0]);
			regulyDobijanie[2 - 1] = GeneratorRegulv3
					.generujRegulyDobijanieRundyKolejne(
							individual, 1)
					.toArray(
							new RegulaAbstrakcyjnaDobijania[0]);
			regulyDobijanie[3 - 1] = GeneratorRegulv3
					.generujRegulyDobijanieRundyKolejne(
							individual, 2)
					.toArray(
							new RegulaAbstrakcyjnaDobijania[0]);
			regulyDobijanie[4 - 1] = GeneratorRegulv3
					.generujRegulyDobijanieRundyKolejne(
							individual, 3)
					.toArray(
							new RegulaAbstrakcyjnaDobijania[0]);

			regulyNaWejscie[1 - 1] = GeneratorRegulv3
					.generujRegulyNaWejscie(individual)
					.toArray(new RegulaAbstrakcyjna[0]);
			regulyNaWejscie[2 - 1] = GeneratorRegulv3
					.generujRegulyNaWejscieRundyKolejne(
							individual, 1)
					.toArray(new RegulaAbstrakcyjna[0]);
			regulyNaWejscie[3 - 1] = GeneratorRegulv3
					.generujRegulyNaWejscieRundyKolejne(
							individual, 2)
					.toArray(new RegulaAbstrakcyjna[0]);
			regulyNaWejscie[4 - 1] = GeneratorRegulv3
					.generujRegulyNaWejscieRundyKolejne(
							individual, 3)
					.toArray(new RegulaAbstrakcyjna[0]);

			regulyIleGrac[1 - 1] = GeneratorRegulv3
					.generujRegulyIleGracRunda1(individual)
					.toArray(
							new RegulaAbstrakcyjnaIleGrac[0]);
			regulyIleGrac[2 - 1] = GeneratorRegulv3
					.generujRegulyIleGracRundyKolejne(
							individual, 1)
					.toArray(
							new RegulaAbstrakcyjnaIleGrac[0]);
			regulyIleGrac[3 - 1] = GeneratorRegulv3
					.generujRegulyIleGracRundyKolejne(
							individual, 2)
					.toArray(
							new RegulaAbstrakcyjnaIleGrac[0]);
			regulyIleGrac[4 - 1] = GeneratorRegulv3
					.generujRegulyIleGracRundyKolejne(
							individual, 3)
					.toArray(
							new RegulaAbstrakcyjnaIleGrac[0]);

		}
	}

	@Override
	final public double play(final int i, final double bid) {


//		
		if (i != 1)
			rezultat = Rezultat.pobierzPrognoze(gra, kolejnosc);
//
//		System.out.println("regula java czy grac "+rundaX_czy_grac(i));
//		System.out.println("regula java stawka "+rundaX_stawka(i));
//		System.out.println("regula java ile grac "+rundaX_ileGrac(rundaX_stawka(i), gra, i));
//		System.out.println("regula java dobijanie "+rundaX_dobijanie(rundaX_stawka(i), i));
//		System.out.flush();
		
		if (!rundaX_czy_grac(i)) {
			return -1.0d;
		} else {

			double stawka = rundaX_stawka(i);

			if (stawka < gra.stawka) {
				if (rundaX_dobijanie(stawka, i))
					stawka = gra.stawka;
				else {
					return -1.0d;
				}
			}

			
//			if (musik > 0) {
//				bilans += musik;
//				musik = 0;
//			}

			double ile = rundaX_ileGrac(stawka, gra, i);
			bilans -= ile - bid;

			// System.out.println(ile);
			return ile;
		}

	}

	static RegulaAbstrakcyjnaIleGrac[] tablicaRegulIleGrac = null;

	public double rundaX_ileGrac(double stawka, Gra gra, final int aRunda) {

		double suma = 0.0d;
		double suma_glosow = 0.0d;

		tablicaRegulIleGrac = regulyIleGrac[aRunda - 1];
		
		float[] wynik_regul;
		for (int i = 0; i < tablicaRegulIleGrac.length; i++) {
			wynik_regul = tablicaRegulIleGrac[i].aplikRegule(gra,
					kolejnosc, individual, rezultat,
					(float) stawka);
//			System.out.println("poczatek "+tablicaRegulIleGrac[i].getPozycjaStartowaWGenotypie());
//			System.out.println("java regula "+(i+1)+" "+wynik_regul[0]+" "+wynik_regul[1]);
			if (wynik_regul[0] != -1.0d) {
				suma_glosow += wynik_regul[1];
				suma += wynik_regul[0] * wynik_regul[1];
			}
//			System.out.println("JAVA regula nr "+(i+1)+" suma "+suma+" suma_glosow "+suma_glosow);
		}
		if (suma_glosow != 0.0d)
			suma = (suma / suma_glosow);

		// System.out.println("suma: "+(suma * stawka) +
		// " stawka "+stawka+" suma_glosow "+suma_glosow);
		// System.out.println(suma * stawka+ " "+suma);

//		System.out.println("java suma "+suma);
//		System.out.flush();
		
		if (suma * stawka < gra.stawka) {
			if (stawka < gra.minimal_bid) {
				return gra.minimal_bid;
			} else {
				return gra.stawka;
			}
		} else {

			if (suma > 1.0) {
				// System.out.println("pleple"+stawka +
				// " "+(suma * stawka));
				return gra.stawka;
			} else {
				// System.out.println("blalba"+stawka +
				// " "+(suma * stawka));
				return gra.stawka + suma
						* (stawka - gra.stawka);

			}
		}

	}

	static RegulaAbstrakcyjna[] tablicaRegul = null;
	static RegulaAbstrakcyjnaDobijania[] tablicaRegulDobijania = null;

	// true, jesli reguly stwierdzily, zeby wziac udzial w licytacji w 1
	// rundzie
	final public boolean rundaX_czy_grac(final int aRunda) {

		int pGlosow = 0;

		tablicaRegul = regulyNaWejscie[aRunda - 1];
		
		final int pWymaganychGlosow = (int) tablicaRegul[0]
				.aplikujRegule(gra, kolejnosc, individual,
						rezultat);
	
		final int pLiczbaRegul = tablicaRegul.length;

		for (int i = 1; i < pLiczbaRegul; i++) {
			pGlosow += tablicaRegul[i].aplikujRegule(gra,
					kolejnosc, individual, rezultat);
		
		}

//		System.out.println(pWymaganychGlosow + " "+pGlosow);
		
		// ostateczna decyzja
		if (pGlosow >= pWymaganychGlosow)
			return true;
		else
			return false;

	}

	final public double rundaX_stawka(final int aRunda) {

		double pStawka = 0.0d;

		
		tablicaRegul = regulyStawkaRunda[aRunda - 1];
	
		final int pLiczbaRegul = tablicaRegul.length;
		
		for (int i = 0; i < pLiczbaRegul; i++) {
			
			pStawka += gra.minimal_bid
					* tablicaRegul[i].aplikujRegule(gra,
							kolejnosc, individual,
							rezultat);
			
		}

		return pStawka;
	}

	final public boolean rundaX_dobijanie(final double aStawka,
			final int aRunda) {

		tablicaRegulDobijania = regulyDobijanie[aRunda - 1];

		final int pLiczbaRegul = tablicaRegulDobijania.length;

		for (int i = 0; i < pLiczbaRegul; i++) {
			if (tablicaRegulDobijania[i].aplikujRegule(gra,
					kolejnosc, aStawka, individual,
					rezultat) == 1.0d) {
				return true;
			}
		

		}

		return false;
	}

}

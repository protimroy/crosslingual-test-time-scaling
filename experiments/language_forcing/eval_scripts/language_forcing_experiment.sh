# Language-Forcing
python src/language_forcing.py --model-name simplescaling/s1.1-32B --device-number 2 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/32B_wait --batch-size 2 --experiment wait
python src/language_forcing.py --model-name simplescaling/s1.1-32B --device-number 2 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/32B_prefix --batch-size 2 --experiment prefix
python src/language_forcing.py --model-name simplescaling/s1.1-32B --device-number 2 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/32B_system --batch-size 2 --experiment system
python src/language_forcing.py --model-name simplescaling/s1.1-32B --device-number 2 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/32B_combined --batch-size 2 --experiment combined
python src/language_forcing.py --model-name simplescaling/s1.1-14B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/14B_wait --batch-size 2 --experiment wait
python src/language_forcing.py --model-name simplescaling/s1.1-14B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/14B_prefix --batch-size 2 --experiment prefix
python src/language_forcing.py --model-name simplescaling/s1.1-14B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/14B_system --batch-size 2 --experiment system
python src/language_forcing.py --model-name simplescaling/s1.1-14B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/14B_combined --batch-size 2 --experiment combined
python src/language_forcing.py --model-name simplescaling/s1.1-7B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/7B_wait --batch-size 2 --experiment wait
python src/language_forcing.py --model-name simplescaling/s1.1-7B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/7B_prefix --batch-size 2 --experiment prefix
python src/language_forcing.py --model-name simplescaling/s1.1-7B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/7B_system --batch-size 2 --experiment system
python src/language_forcing.py --model-name simplescaling/s1.1-7B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/7B_combined --batch-size 2 --experiment combined

# NxN Experiment
python src/language_forcing_nxn.py --model-name simplescaling/s1.1-14B --device-number 1 --lang-subsets bn,de,en,es,fr,ja,ru,sw,te,th,zh --output-dir processed/14B_NxN --batch-size 2 --experiment combined
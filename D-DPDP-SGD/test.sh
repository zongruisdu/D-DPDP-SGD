for i in {0..4};do
python3 worker.py -rank $i -world_size 5&
done
echo "END"

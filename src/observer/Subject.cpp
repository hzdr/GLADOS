#include <ddrf/observer/Subject.h>

namespace ddrf
{
	auto Subject::attach(Observer* o) -> void
	{
		observers_.push_back(o);
	}

	auto Subject::detach(Observer* o) -> void
	{
		observers_.remove(o);
	}

	auto Subject::notify() -> void
	{
		for(auto& o : observers_)
			o->update(this);
	}
}
